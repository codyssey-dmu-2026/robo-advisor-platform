"""RL 서빙 서버 (Port 8002): rl-engine에서 학습한 PPO 모델로 자산 비중을 계산한다.

서버가 켜질 때 평가 구간 전체를 모델로 한 번 재생해 날짜별 목표 비중을 메모리에 올리고,
요청이 오면 그 표에서 찾아 돌려준다. rl-engine의 환경 코드를 그대로 쓰므로
백테스트(eval_test_daily.csv)와 같은 비중이 나온다.
모델·데이터는 환경 변수로 지정하므로 RL_MODEL_NAME만 바꾸면 모델을 교체할 수 있다.
"""

import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")

_SIBLINGS = Path(__file__).resolve().parent.parent
# rl-engine 레포 경로 (기본값: platform과 나란히 있는 형제 폴더)
RL_REPO_PATH = Path(os.getenv("RL_REPO_PATH", _SIBLINGS / "rl-engine"))
# 모델 파일 폴더, 사용할 모델 이름(.zip 제외), data-pipeline 결과 폴더
RL_MODELS_DIR = Path(os.getenv("RL_MODELS_DIR", RL_REPO_PATH / "models"))
RL_MODEL_NAME = os.getenv("RL_MODEL_NAME", "ppo_test_w4_simple_s42")
RL_DATA_DIR = Path(
    os.getenv("RL_DATA_DIR", _SIBLINGS / "data-pipeline" / "data" / "processed")
)
# rl-engine Walk-Forward 백테스트 결과 CSV 폴더 (eval_test_*.csv, 모델 번들에 포함)
RL_RESULTS_DIR = Path(os.getenv("RL_RESULTS_DIR", RL_MODELS_DIR))

TRADING_DAYS = 252
MDD_LIMIT = 0.15  # Safe-Guard 한도
BENCHMARK = "SPY"  # rl-engine evaluate.py 기본 벤치마크와 동일
# rl-engine CSV 컬럼 → Gateway·대시보드 지표 키 (명세 12개 지표)
METRIC_COLUMNS = {
    "cum_return": "cumulative_return",
    "cagr": "cagr",
    "ann_vol": "annualized_volatility",
    "var_95": "var_95",
    "cvar_95": "cvar_95",
    "max_mdd": "mdd",
    "sharpe": "sharpe",
    "sortino": "sortino",
    "calmar": "calmar",
    "alpha": "alpha",
    "beta": "beta",
    "information_ratio": "information_ratio",
}

DISCLAIMER = (
    "본 결과는 교육 목적이며 실제 투자 조언에 사용할 수 없습니다. "
    "백테스팅 성과는 미래 수익을 보장하지 않습니다."
)
ONE_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class PolicySchedule:
    """모델이 거래일 종가 이후 정한 목표 비중 표 (dates[i]의 결정 = weights[i])."""

    model_name: str
    tickers: List[str]
    dates: pd.DatetimeIndex
    weights: np.ndarray

    @property
    def as_of_min(self) -> pd.Timestamp:
        return self.dates[0] + ONE_DAY

    @property
    def as_of_max(self) -> pd.Timestamp:
        return self.dates[-1] + ONE_DAY

    def index_for(self, as_of: Optional[str]) -> int:
        """as_of 시각에 이미 끝난 마지막 거래일의 위치 (미래 정보 누수 방지)."""
        if as_of is None:
            return len(self.dates) - 1
        try:
            when = pd.Timestamp(as_of).tz_localize(None)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, f"as_of 형식이 잘못되었습니다: {as_of}") from error
        # 거래일 d의 데이터는 d가 끝난 뒤(d + 1일 0시)부터 사용할 수 있다
        index = int(self.dates.searchsorted(when - ONE_DAY, side="right")) - 1
        if index < 0 or when.normalize() > self.as_of_max:
            raise HTTPException(
                422,
                f"as_of는 {self.as_of_min.date()} ~ {self.as_of_max.date()} "
                "범위여야 합니다 (모델의 평가 구간)",
            )
        return index


def _build_schedule() -> PolicySchedule:
    """모델과 데이터를 읽어 평가 구간 전체의 날짜별 목표 비중을 계산한다."""
    source = RL_REPO_PATH / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"rl-engine 코드를 찾을 수 없습니다: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    # torch를 쓰는 RL 전용 이미지에서만 필요하므로 여기서 불러온다
    from stable_baselines3 import PPO

    from data_loader import load_walk_forward_window
    from environment import action_to_weights
    from model_meta import check_compatible, load_meta
    from rollout import build_test_env

    path = RL_MODELS_DIR / f"{RL_MODEL_NAME}.zip"
    if not path.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    meta = load_meta(path)
    if meta["stage"] != "test":
        raise ValueError(f"test 단계 모델만 서빙합니다: {meta['stage']}")

    data = load_walk_forward_window(meta["window"], RL_DATA_DIR)
    env = build_test_env(data, meta["lookback"])
    check_compatible(meta, data, env)  # 자산 순서·입력 크기가 다르면 여기서 중단
    model = PPO.load(str(path), device="cpu")

    dates, weights = [], []
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        dates.append(env.dates[env.current_step])
        weights.append(action_to_weights(action, meta["max_weight"]))
        if env.current_step >= env.n_steps - 1:  # 마지막 날은 결정만 하고 끝
            break
        obs, *_ = env.step(action)

    return PolicySchedule(
        model_name=RL_MODEL_NAME,
        tickers=list(data.tickers),
        dates=pd.DatetimeIndex(dates).tz_localize(None),
        weights=np.asarray(weights, dtype=float),
    )


@lru_cache(maxsize=1)
def get_schedule() -> PolicySchedule:
    """최초 1회만 계산한다. 실패하면 저장하지 않으므로 다음 호출에서 다시 시도한다."""
    return _build_schedule()


def _clean(value) -> Optional[float]:
    """NaN·무한대는 JSON에 넣을 수 없고 0으로 바꾸면 왜곡되므로 None으로 둔다."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _value_curve(returns: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """시작 자산 1.0을 고점 후보에 넣은 자산가치와 낙폭 (rl-engine max_drawdown과 동일)."""
    value = np.cumprod(1.0 + returns.to_numpy(dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], value]))[1:]
    return value, 1.0 - value / peak


def _series_metrics(returns: pd.Series) -> Dict[str, Optional[float]]:
    """벤치마크 자체의 지표. rl-engine metrics.compute_metrics와 같은 공식(rf=0)."""
    r = returns.astype(float)
    n = len(r)
    cum = float(np.prod(1.0 + r) - 1.0)
    cagr = (1.0 + cum) ** (TRADING_DAYS / n) - 1.0 if cum > -1.0 else -1.0
    std = float(r.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
    mdd = float(_value_curve(r)[1].max())
    q = float(np.quantile(r, 0.05))
    return {
        "cumulative_return": _clean(cum),
        "cagr": _clean(cagr),
        "annualized_volatility": _clean(std * np.sqrt(TRADING_DAYS)),
        "var_95": _clean(-q),
        "cvar_95": _clean(-r[r <= q].mean()),
        "mdd": _clean(mdd),
        "sharpe": _clean(r.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else None),
        "sortino": _clean(
            r.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else None
        ),
        "calmar": _clean(cagr / mdd if mdd > 0 else None),
        # 자기 자신과 비교하므로 알파 0, 베타 1, 정보비율은 정의되지 않음
        "alpha": 0.0,
        "beta": 1.0,
        "information_ratio": None,
    }


def _row_metrics(row: pd.Series) -> Dict[str, Optional[float]]:
    return {key: _clean(row[col]) for col, key in METRIC_COLUMNS.items()}


def _breaches(dates: List[str], drawdown: np.ndarray) -> List[dict]:
    """낙폭이 Safe-Guard 한도를 넘은 구간 (시작일, 마지막일, 일수, 최대 낙폭)."""
    events, start = [], None
    for i, dd in enumerate(np.append(drawdown, 0.0)):
        if dd > MDD_LIMIT and start is None:
            start = i
        elif dd <= MDD_LIMIT and start is not None:
            events.append(
                {
                    "start": dates[start],
                    "end": dates[i - 1],
                    "days": i - start,
                    "max_drawdown": _clean(drawdown[start:i].max()),
                }
            )
            start = None
    return events


def _schedule_weights() -> Optional[dict]:
    """현재 모델의 날짜별 목표 비중 (모델을 못 쓰면 None)."""
    try:
        schedule = get_schedule()
    except Exception:
        return None
    return {
        "dates": [str(d.date()) for d in schedule.dates],
        "tickers": schedule.tickers,
        "values": np.round(schedule.weights, 6).tolist(),
    }


def _build_performance() -> dict:
    """현재 모델(RL_MODEL_NAME)과 같은 설정으로 구간마다 재학습한 Walk-Forward 성과."""
    meta = json.loads(
        (RL_MODELS_DIR / f"{RL_MODEL_NAME}.json").read_text(encoding="utf-8")
    )
    strategy, seed = f"PPO-{meta['variant']}", str(meta["seed"])

    def read(name: str) -> pd.DataFrame:
        frame = pd.read_csv(
            RL_RESULTS_DIR / f"eval_test_{name}.csv", dtype={"seed": str}
        )
        return frame.assign(seed=frame["seed"].fillna("-"))

    daily, windows, walkforward = read("daily"), read("windows"), read("walkforward")
    mine = daily[(daily.strategy == strategy) & (daily.seed == seed)].sort_values(
        "date"
    )
    if mine.empty:
        raise ValueError(f"결과 CSV에 {strategy} (시드 {seed}) 기록이 없습니다")
    dates = mine["date"].tolist()
    equal = (
        daily[daily.strategy == "EqualWeight"]
        .set_index("date")["net_return"]
        .loc[dates]
    )
    bench = mine.set_index("date")["benchmark_return"]
    curves = {
        "model": mine.set_index("date")["net_return"],
        "equal_weight": equal,
        "benchmark": bench,
    }
    series = {"dates": dates}
    for key, returns in curves.items():
        value, drawdown = _value_curve(returns)
        series[key] = {
            "cumulative_return": np.round(value - 1.0, 6).tolist(),
            "drawdown": np.round(drawdown, 6).tolist(),
        }

    periods = (
        mine.groupby("window")["date"]
        .agg(["min", "max"])
        .reset_index()
        .to_dict("records")
    )

    def window_rows(frame: pd.DataFrame) -> List[dict]:
        mean = frame.groupby("window")[list(METRIC_COLUMNS)].mean()
        return [
            {"window": int(w), "metrics": _row_metrics(row)}
            for w, row in mean.iterrows()
        ]

    def entry(name, seed_label, kind, total, frame):
        return {
            "strategy": name,
            "seed": seed_label,
            "kind": kind,
            "total": total,
            "windows": window_rows(frame),
        }

    def runs(name, seed_value=None):
        """(4년 전체 지표, 구간별 지표) 행. 시드를 주지 않으면 모든 시드."""

        def pick(frame):
            mask = frame.strategy == name
            return (
                frame[mask & (frame.seed == seed_value)] if seed_value else frame[mask]
            )

        return pick(walkforward), pick(windows)

    total, per_window = runs(strategy, seed)
    strategies = [
        entry(strategy, seed, "model", _row_metrics(total.iloc[0]), per_window)
    ]
    total, per_window = runs("EqualWeight")
    strategies.append(
        entry("EqualWeight", "-", "baseline", _row_metrics(total.iloc[0]), per_window)
    )
    bench_windows = [
        {
            "window": int(w),
            "metrics": _series_metrics(bench[mine.set_index("date")["window"] == w]),
        }
        for w in sorted(mine["window"].unique())
    ]
    strategies.append(
        {
            "strategy": BENCHMARK,
            "seed": "-",
            "kind": "benchmark",
            "total": _series_metrics(bench),
            "windows": bench_windows,
        }
    )
    # 다른 보상 함수 변형은 시드 평균으로 비교 (특정 시드만 고르는 편향 방지)
    for name in sorted(set(walkforward.strategy) - {strategy, "EqualWeight"}):
        total, per_window = runs(name)
        mean = total[list(METRIC_COLUMNS)].mean()
        strategies.append(
            entry(name, "mean", "variant", _row_metrics(mean), per_window)
        )

    return {
        "status": "ok",
        "data_mode": "real",
        "model_name": RL_MODEL_NAME,
        "strategy": strategy,
        "seed": seed,
        "benchmark": BENCHMARK,
        "source": "rl-engine Walk-Forward 테스트 (구간마다 재학습, 비용 차감 후)",
        "windows": [
            {"window": int(p["window"]), "start": p["min"], "end": p["max"]}
            for p in periods
        ],
        "strategies": strategies,
        "series": series,
        "breaches": _breaches(dates, np.array(series["model"]["drawdown"])),
        "mdd_limit": MDD_LIMIT,
        "disclaimer": DISCLAIMER,
    }


@lru_cache(maxsize=1)
def get_performance() -> dict:
    return _build_performance()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 첫 요청이 5초 제한을 넘지 않도록 시작할 때 미리 계산한다
    try:
        schedule = get_schedule()
        logger.info("모델 로드 완료: %s", schedule.model_name)
    except Exception:
        logger.exception("모델 로드 실패: /optimize는 503을 반환합니다")
    yield


app = FastAPI(
    title="RL Portfolio Optimization Server",
    version="1.0.0",
    description="PPO 강화학습 모델 기반 자산 배분 추론 서버 (Port 8002)",
    lifespan=lifespan,
)


class OptimizeRequest(BaseModel):
    tickers: List[str] = Field(
        default=[
            "000660",
            "005930",
            "069500",
            "148070",
            "AAPL",
            "GLD",
            "MSFT",
            "NVDA",
            "QQQ",
            "SPY",
            "TLT",
            "VNQ",
        ]
    )
    as_of: Optional[str] = Field(default=None, description="기준 시각. 없으면 최신")
    risk_override: Optional[bool] = Field(default=True)


@app.get("/health")
def health_check():
    base = {
        "service": "rl-engine",
        "version": app.version,
        "is_mock": False,
        "safeguard_mdd_limit": 0.15,
    }
    try:
        schedule = get_schedule()
    except Exception as error:
        return {
            **base,
            "status": "degraded",
            "model_loaded": False,
            "reason": str(error),
        }
    return {
        **base,
        "status": "ok",
        "model_loaded": True,
        "model_name": schedule.model_name,
        "as_of_min": str(schedule.as_of_min.date()),
        "as_of_max": str(schedule.as_of_max.date()),
    }


@app.get("/performance")
def performance():
    """현재 모델의 Walk-Forward 백테스트 성과 (동일가중·SPY·다른 보상 함수와 비교)."""
    try:
        result = get_performance()
    except Exception as error:
        logger.exception("백테스트 결과 로드 실패")
        raise HTTPException(
            503, f"백테스트 결과를 읽을 수 없습니다: {error}"
        ) from error
    return {**result, "weights": _schedule_weights()}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    try:
        schedule = get_schedule()
    except Exception as error:
        logger.exception("모델 로드 실패")
        raise HTTPException(503, f"모델을 사용할 수 없습니다: {error}") from error

    if sorted(req.tickers) != sorted(schedule.tickers):
        raise HTTPException(
            422,
            f"이 모델은 {len(schedule.tickers)}개 자산 전체만 지원합니다: "
            f"{', '.join(schedule.tickers)}",
        )

    index = schedule.index_for(req.as_of)
    by_ticker = dict(zip(schedule.tickers, schedule.weights[index]))
    weights: Dict[str, float] = {t: round(float(by_ticker[t]), 6) for t in req.tickers}

    return {
        "status": "ok",
        "service": "rl-engine",
        "is_mock": False,
        "model_used": schedule.model_name,
        "as_of": req.as_of,
        "decision_date": str(schedule.dates[index].date()),
        "weights": weights,
        "transaction_cost_bps": 1.5,
        "slippage_bps": 5.0,
        "safeguard_triggered": False,
        # 현재 모델은 RiskTag 없이 학습되어 risk_override를 비중에 반영할 수 없다
        "risk_override_applied": False,
        "message": (
            f"PPO 정책({schedule.model_name})의 목표 비중 "
            "(수수료 0.015%, 슬리피지 0.05% 명세값으로 학습·평가)"
        ),
        "disclaimer": DISCLAIMER,
    }
