"""RL 서빙 서버 (Port 8002): rl-engine에서 학습한 PPO 모델로 자산 비중을 계산한다.

서버가 켜질 때 평가 구간 전체를 모델로 한 번 재생해 날짜별 목표 비중을 메모리에 올리고,
요청이 오면 그 표에서 찾아 돌려준다. rl-engine의 환경 코드를 그대로 쓰므로
백테스트(eval_test_daily.csv)와 같은 비중이 나온다.
모델·데이터는 환경 변수로 지정하므로 RL_MODEL_NAME만 바꾸면 모델을 교체할 수 있다.
RL_MODEL_URI(예: models:/ppo-portfolio@production)를 주면 MLflow 모델 저장소에서
그 별칭이 가리키는 버전을 내려받아 서빙한다.

리스크 태그 Safe-Guard: agentic-research가 만든 위험 이벤트(RiskTag)를 읽어, 위험 신호가
강한 자산의 비중을 줄인 뒤 응답한다 (과제의 "리스크 모니터링 모듈 연동"). 모델을 다시
학습하지 않는 서빙 단계 규칙이므로 /performance의 백테스트 수치에는 반영되지 않는다.
"""

import json
import logging
import math
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
# MLflow 모델 저장소의 모델 URI. 비우면 위 파일(RL_MODELS_DIR/RL_MODEL_NAME)을 쓴다
RL_MODEL_URI = os.getenv("RL_MODEL_URI", "").strip()
# MLflow에서 내려받은 체크포인트를 둘 폴더
RL_MODEL_CACHE = Path(
    os.getenv("RL_MODEL_CACHE", Path(tempfile.gettempdir()) / "rl-model-registry")
)
RL_DATA_DIR = Path(
    os.getenv("RL_DATA_DIR", _SIBLINGS / "data-pipeline" / "data" / "processed")
)
# rl-engine Walk-Forward 백테스트 결과 CSV 폴더 (eval_test_*.csv, 모델 번들에 포함)
RL_RESULTS_DIR = Path(os.getenv("RL_RESULTS_DIR", RL_MODELS_DIR))
# 학습 곡선 CSV (rl-engine TensorBoard 기록을 scripts/export_learning_curves.py로 변환)
RL_LEARNING_CURVES = Path(
    os.getenv(
        "RL_LEARNING_CURVES",
        Path(__file__).resolve().parent / "data" / "learning_curves.csv",
    )
)
# agentic-research가 만든 리스크 태그 파일 (읽기 전용)
RISK_TAGS_PATH = Path(
    os.getenv(
        "RISK_TAGS_PATH",
        _SIBLINGS / "agentic-research" / "artifacts" / "risk_tags.jsonl",
    )
)
# 위험 점수(심각도 × 신뢰도)가 이 값 이상인 자산만 비중을 줄인다
RISK_TAG_THRESHOLD = float(os.getenv("RISK_TAG_THRESHOLD", "0.5"))
# 축소 비율 = RISK_TAG_MAX_CUT × 위험 점수 (점수 1.0이면 비중의 절반을 줄임)
RISK_TAG_MAX_CUT = float(os.getenv("RISK_TAG_MAX_CUT", "0.5"))

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
    max_weight: float = 0.4  # 자산당 비중 상한 (모델 학습 설정)

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


_REGISTRY: dict = {}  # MLflow에서 한 번 찾은 모델 (서버가 떠 있는 동안 고정)


def model_source() -> dict:
    """서빙할 모델의 이름·파일 폴더와 출처(file / mlflow).

    MLflow에 연결하지 못하면 파일 모델로 대체하고 사유를 남긴다(서빙이 멈추지 않게).
    """
    if not RL_MODEL_URI:
        return {"name": RL_MODEL_NAME, "dir": RL_MODELS_DIR, "source": "file"}
    if not _REGISTRY:
        try:
            from mlops.registry import resolve_model

            _REGISTRY.update(resolve_model(RL_MODEL_URI, RL_MODEL_CACHE))
            logger.info(
                "MLflow 모델 사용: %s → %s v%s",
                RL_MODEL_URI,
                _REGISTRY["name"],
                _REGISTRY["version"],
            )
        except Exception as error:
            logger.warning("MLflow 모델을 쓸 수 없어 파일 모델로 대체 (%s)", error)
            _REGISTRY.update(
                {
                    "name": RL_MODEL_NAME,
                    "dir": RL_MODELS_DIR,
                    "source": "file",
                    "uri": RL_MODEL_URI,
                    "fallback_reason": f"{type(error).__name__}: {error}",
                }
            )
    return dict(_REGISTRY)


def model_source_info() -> dict:
    """/health용: 폴더 경로는 빼고 출처 정보만."""
    return {k: str(v) for k, v in model_source().items() if k != "dir"}


@dataclass(frozen=True)
class Policy:
    """학습된 모델과 rl-engine 평가 환경. 같은 환경으로 비중표·백테스트를 재생한다."""

    model_name: str
    tickers: List[str]
    max_weight: float
    model: Any  # predict(obs, deterministic=True) -> (action, state)
    make_env: Callable[[], Any]  # rl-engine 테스트 환경 (reset·step·dates·current_step)
    action_to_weights: Callable[[Any, float], np.ndarray]
    weights_to_action: Callable[[np.ndarray], Any]


def _naive(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tzinfo else stamp


def replay(
    policy: Policy,
    adjust: Optional[Callable[[pd.Timestamp, np.ndarray], np.ndarray]] = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame]:
    """평가 구간을 하루씩 재생한다. (결정일, 목표 비중, 일별 성과)

    adjust를 주면 모델 비중을 그 함수로 바꾼 뒤 체결한다. 바뀐 비중이 다음 날
    관측(현재 보유 비중)에 들어가므로 Safe-Guard를 실제로 켠 것과 같은 폐루프 재생이다.
    거래비용·체결 규칙은 rl-engine 환경 그대로다.
    """
    env = policy.make_env()
    obs, _ = env.reset(seed=42)
    dates, weights, rows = [], [], []
    while True:
        action, _ = policy.model.predict(obs, deterministic=True)
        date = _naive(env.dates[env.current_step])
        target = policy.action_to_weights(action, policy.max_weight)
        if adjust is not None:
            target = adjust(date, target)
            action = policy.weights_to_action(target)
        dates.append(date)
        weights.append(target)
        if env.current_step >= env.n_steps - 1:  # 마지막 날은 결정만 하고 끝
            break
        obs, _, _, _, info = env.step(action)
        rows.append(
            {
                "date": _naive(env.dates[env.current_step]),
                "net_return": float(info["net_return"]),
                "turnover": float(info["turnover"]),
            }
        )
    daily = pd.DataFrame(rows, columns=["date", "net_return", "turnover"])
    return (
        pd.DatetimeIndex(dates),
        np.asarray(weights, dtype=float),
        daily.set_index("date"),
    )


def _load_policy() -> Policy:
    """모델과 데이터를 읽어 평가 환경을 준비한다."""
    source = RL_REPO_PATH / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"rl-engine 코드를 찾을 수 없습니다: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    # torch를 쓰는 RL 전용 이미지에서만 필요하므로 여기서 불러온다
    from stable_baselines3 import PPO

    from data_loader import load_walk_forward_window
    from environment import action_to_weights, weights_to_action
    from model_meta import check_compatible, load_meta
    from rollout import build_test_env

    source = model_source()
    path = source["dir"] / f"{source['name']}.zip"
    if not path.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    meta = load_meta(path)
    if meta["stage"] != "test":
        raise ValueError(f"test 단계 모델만 서빙합니다: {meta['stage']}")

    data = load_walk_forward_window(meta["window"], RL_DATA_DIR)
    env = build_test_env(data, meta["lookback"])
    check_compatible(meta, data, env)  # 자산 순서·입력 크기가 다르면 여기서 중단
    model = PPO.load(str(path), device="cpu")
    return Policy(
        model_name=source["name"],
        tickers=list(data.tickers),
        max_weight=float(meta["max_weight"]),
        model=model,
        make_env=lambda: build_test_env(data, meta["lookback"]),
        action_to_weights=action_to_weights,
        weights_to_action=weights_to_action,
    )


@lru_cache(maxsize=1)
def get_policy() -> Policy:
    """최초 1회만 읽는다. 실패하면 저장하지 않으므로 다음 호출에서 다시 시도한다."""
    return _load_policy()


def _build_schedule() -> PolicySchedule:
    """평가 구간 전체의 날짜별 목표 비중 (모델 비중 그대로 재생)."""
    policy = get_policy()
    dates, weights, _ = replay(policy)
    return PolicySchedule(
        model_name=policy.model_name,
        tickers=policy.tickers,
        dates=dates,
        weights=weights,
        max_weight=policy.max_weight,
    )


@lru_cache(maxsize=1)
def get_schedule() -> PolicySchedule:
    """최초 1회만 계산한다. 실패하면 저장하지 않으므로 다음 호출에서 다시 시도한다."""
    return _build_schedule()


# ---------------- 리스크 태그 Safe-Guard ----------------
TAG_COLUMNS = [
    "tag_id",
    "asset_id",
    "risk_type",
    "severity",
    "confidence",
    "available_at",
    "expires_at",
    "source_url",
]


@lru_cache(maxsize=1)
def get_risk_tags() -> pd.DataFrame:
    """리스크 태그 표. 파일이 없으면 예외가 나고 저장되지 않아 다음 호출에서 다시 읽는다."""
    with RISK_TAGS_PATH.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    frame = pd.DataFrame(rows, columns=TAG_COLUMNS)
    for column in ("available_at", "expires_at"):
        # 한국·미국 시간대가 섞여 있으므로 UTC로 통일해 비교한다
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
    # 에이전트의 RiskTagProvider.vector()와 같은 정의: 심각도 × 신뢰도
    frame["score"] = frame["severity"].astype(float) * frame["confidence"].astype(float)
    return frame


def _active_tags(
    tags: pd.DataFrame, decision_date: pd.Timestamp, tickers: List[str]
) -> pd.DataFrame:
    """결정일 장 마감까지 공개됐고 아직 만료되지 않은 태그 (미래 정보 누수 방지)."""
    # 결정일이 끝나는 시각(다음 날 0시 UTC). 한국·미국 장 마감이 모두 이 시각 이전이다
    at = (decision_date + ONE_DAY).tz_localize("UTC")
    live = tags[
        (tags["available_at"] <= at)
        & (at < tags["expires_at"])
        & tags["asset_id"].isin(tickers)
    ]
    return live.sort_values(["score", "tag_id"], ascending=[False, True])


def _spread(
    weights: np.ndarray, amount: float, receivers: np.ndarray, max_weight: float
) -> tuple[np.ndarray, float]:
    """amount를 receivers에게 현재 비중 비율대로 나눠 주되 자산당 상한을 넘기지 않는다."""
    weights, receivers = weights.copy(), receivers.copy()
    while amount > 1e-12:
        receivers &= weights < max_weight - 1e-12
        if not receivers.any():
            break
        base = weights[receivers]
        share = (
            base / base.sum() if base.sum() > 0 else np.full(base.size, 1 / base.size)
        )
        give = np.minimum(amount * share, max_weight - base)
        weights[receivers] += give
        amount -= float(give.sum())
    return weights, amount


def apply_risk_tags(
    weights: np.ndarray, scores: np.ndarray, max_weight: float
) -> np.ndarray:
    """위험 점수가 기준 이상인 자산의 비중을 줄이고, 줄인 만큼을 나머지 자산에 나눈다.

    축소 비율은 RISK_TAG_MAX_CUT × 점수. 줄인 비중은 위험 신호가 없는 자산이 원래 비율대로
    받고, 그 자산들이 상한에 걸려 다 받지 못할 때만 남은 만큼을 전체 자산에 돌려준다.
    """
    weights = np.asarray(weights, dtype=float)
    cut = np.where(scores >= RISK_TAG_THRESHOLD, RISK_TAG_MAX_CUT * scores, 0.0)
    adjusted = weights * (1.0 - cut)
    freed = float(weights.sum() - adjusted.sum())
    for receivers in (cut == 0, np.ones(len(weights), dtype=bool)):
        adjusted, freed = _spread(adjusted, freed, receivers, max_weight)
    return adjusted


def _tag_records(live: pd.DataFrame) -> List[dict]:
    return [
        {
            "tag_id": row.tag_id,
            "asset_id": row.asset_id,
            "risk_type": row.risk_type,
            "severity": _clean(row.severity),
            "confidence": _clean(row.confidence),
            "score": round(float(row.score), 4),
            "available_at": row.available_at.isoformat(),
            "expires_at": row.expires_at.isoformat(),
            "source_url": row.source_url,
            "applied": bool(row.score >= RISK_TAG_THRESHOLD),
        }
        for row in live.itertuples()
    ]


def _scores(live: pd.DataFrame, tickers: List[str]) -> np.ndarray:
    """자산별 위험 점수: 유효한 태그 중 최대 (심각도 × 신뢰도)."""
    best = live.groupby("asset_id")["score"].max()
    return np.array([float(best.get(t, 0.0)) for t in tickers])


def guard_decision(schedule: PolicySchedule, index: int, tags: pd.DataFrame) -> dict:
    """index번째 결정일의 Safe-Guard 결과: 조정된 비중, 근거 태그, 자산별 조정 내역."""
    raw = schedule.weights[index]
    live = _active_tags(tags, schedule.dates[index], schedule.tickers)
    scores = _scores(live, schedule.tickers)
    adjusted = apply_risk_tags(raw, scores, schedule.max_weight)
    adjustments = [
        {
            "asset": ticker,
            "score": round(float(scores[i]), 4),
            "before": round(float(raw[i]), 6),
            "after": round(float(adjusted[i]), 6),
        }
        for i, ticker in enumerate(schedule.tickers)
        if scores[i] >= RISK_TAG_THRESHOLD
    ]
    return {
        "weights": adjusted,
        "tags": _tag_records(live),
        "adjustments": sorted(adjustments, key=lambda a: a["after"] - a["before"]),
        # 위험 자산에서 실제로 빠져나간 비중 합계
        "moved_weight": round(sum(a["before"] - a["after"] for a in adjustments), 6),
    }


def _guard_rule(schedule: PolicySchedule) -> dict:
    return {
        "score": "유효한 태그 중 최대 (심각도 × 신뢰도)",
        "threshold": RISK_TAG_THRESHOLD,
        "max_cut": RISK_TAG_MAX_CUT,
        "max_weight": schedule.max_weight,
    }


def build_guard_summary(schedule: PolicySchedule, tags: pd.DataFrame) -> dict:
    """현재 모델의 평가 구간 전체에서 Safe-Guard가 언제, 얼마나 작동했는지 요약."""
    days, seen, by_asset = [], {}, {}
    for index, date in enumerate(schedule.dates):
        result = guard_decision(schedule, index, tags)
        for tag in result["tags"]:
            seen[tag["tag_id"]] = tag
        if not result["adjustments"]:
            continue
        days.append(
            {
                "date": str(date.date()),
                # 이 결정이 적용되는 /optimize 기준일
                "as_of": str((date + ONE_DAY).date()),
                "moved_weight": result["moved_weight"],
                "assets": [a["asset"] for a in result["adjustments"]],
            }
        )
        for item in result["adjustments"]:
            entry = by_asset.setdefault(
                item["asset"], {"asset": item["asset"], "days": 0, "moved_weight": 0.0}
            )
            entry["days"] += 1
            entry["moved_weight"] += item["before"] - item["after"]
    moved = [d["moved_weight"] for d in days]
    for entry in by_asset.values():
        # 발동일 하루 평균으로 줄어든 비중
        entry["moved_weight"] = round(entry["moved_weight"] / entry["days"], 6)
    return {
        "status": "ok",
        "model_name": schedule.model_name,
        "rule": _guard_rule(schedule),
        "tag_count": int(len(tags)),
        "period": {
            "start": str(schedule.dates[0].date()),
            "end": str(schedule.dates[-1].date()),
        },
        "trading_days": len(schedule.dates),
        "adjusted_days": len(days),
        "average_moved": round(float(np.mean(moved)), 6) if moved else 0.0,
        "max_moved": round(float(np.max(moved)), 6) if moved else 0.0,
        "days": days,
        "by_asset": sorted(by_asset.values(), key=lambda e: -e["days"]),
        "tags": sorted(seen.values(), key=lambda t: t["available_at"]),
        "limitation": (
            "서빙 단계에서 적용하는 규칙입니다. 모델은 리스크 태그 없이 학습됐고, "
            "강화학습 성과 화면의 백테스트 성과에는 이 조정이 반영되지 않았습니다. "
            "끔/켬 비교는 리스크 모니터링 화면의 백테스트 비교에서 볼 수 있습니다."
        ),
        "disclaimer": DISCLAIMER,
    }


_GUARD_SUMMARY: list = []  # [(비중표, 태그 표, 요약)] 마지막 계산 1건


def get_guard_summary(schedule: PolicySchedule, tags: pd.DataFrame) -> dict:
    """같은 비중표·태그로는 한 번만 계산한다 (전체 구간 계산에 0.5초가량 걸림)."""
    if _GUARD_SUMMARY:
        cached_schedule, cached_tags, summary = _GUARD_SUMMARY[0]
        if cached_schedule is schedule and cached_tags is tags:
            return summary
    summary = build_guard_summary(schedule, tags)
    _GUARD_SUMMARY[:] = [(schedule, tags, summary)]
    return summary


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
    """현재 모델과 같은 설정으로 구간마다 재학습한 Walk-Forward 성과."""
    source = model_source()
    model_name = source["name"]
    meta = json.loads(
        (source["dir"] / f"{model_name}.json").read_text(encoding="utf-8")
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
        "model_name": model_name,
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


def _published_daily() -> Optional[pd.Series]:
    """rl-engine이 공개한 현재 모델의 일별 수익률 (재생 결과 대조용). 없으면 None."""
    try:
        meta = json.loads(
            (model_source()["dir"] / f"{model_source()['name']}.json").read_text(
                encoding="utf-8"
            )
        )
        daily = pd.read_csv(RL_RESULTS_DIR / "eval_test_daily.csv", dtype={"seed": str})
    except (OSError, ValueError, KeyError):
        return None
    mine = daily[
        (daily.strategy == f"PPO-{meta.get('variant')}")
        & (daily.seed == str(meta.get("seed")))
        & (daily.window == meta.get("window"))
    ]
    if mine.empty:
        return None
    return mine.set_index(pd.to_datetime(mine["date"]))["net_return"]


def build_guard_backtest(policy: Policy, tags: pd.DataFrame) -> dict:
    """같은 모델·같은 환경으로 리스크 태그 Safe-Guard 끔/켬을 재생해 성과를 비교한다."""
    moved: Dict[pd.Timestamp, float] = {}

    def guarded(date: pd.Timestamp, weights: np.ndarray) -> np.ndarray:
        live = _active_tags(tags, date, policy.tickers)
        adjusted = apply_risk_tags(
            weights, _scores(live, policy.tickers), policy.max_weight
        )
        cut = float(np.clip(weights - adjusted, 0.0, None).sum())
        if cut > 1e-9:
            moved[date] = cut
        return adjusted

    _, _, off = replay(policy)
    _, _, on = replay(policy, adjust=guarded)

    def metrics(frame: pd.DataFrame) -> dict:
        result = _series_metrics(frame["net_return"])
        for key in ("alpha", "beta", "information_ratio"):  # 벤치마크 비교 지표는 제외
            result.pop(key)
        result["avg_turnover"] = _clean(frame["turnover"].mean())
        return result

    model, guard = metrics(off), metrics(on)
    series = {"dates": [str(d.date()) for d in off.index]}
    for key, frame in (("model", off), ("guarded", on)):
        value, drawdown = _value_curve(frame["net_return"])
        series[key] = {
            "cumulative_return": np.round(value - 1.0, 6).tolist(),
            "drawdown": np.round(drawdown, 6).tolist(),
        }

    # 끈 쪽 재생이 rl-engine 공개 결과와 같은지 (같은 모델·데이터인지 확인)
    published = _published_daily()
    check = None
    if published is not None:
        common = off.index.intersection(published.index)
        check = {
            "matched_days": int(len(common)),
            "max_abs_diff": _clean(
                (off.loc[common, "net_return"] - published.loc[common]).abs().max()
            ),
        }

    return {
        "status": "ok",
        "model_name": policy.model_name,
        "period": {"start": series["dates"][0], "end": series["dates"][-1]},
        "trading_days": len(off),
        "method": (
            "폐루프 재생: Safe-Guard로 바꾼 비중으로 체결하고, 그 비중이 다음 날 "
            "모델 입력(현재 보유 비중)에 들어간다. 거래비용·체결은 rl-engine 환경과 동일"
        ),
        "rule": {
            "score": "유효한 태그 중 최대 (심각도 × 신뢰도)",
            "threshold": RISK_TAG_THRESHOLD,
            "max_cut": RISK_TAG_MAX_CUT,
            "max_weight": policy.max_weight,
        },
        "adjusted_days": len(moved),
        "average_moved": _clean(np.mean(list(moved.values()))) if moved else 0.0,
        "metrics": {"model": model, "guarded": guard},
        "difference": {
            key: (
                None
                if model[key] is None or guard[key] is None
                else _clean(guard[key] - model[key])
            )
            for key in model
        },
        "series": series,
        "reproduces_published": check,
        "limitation": (
            "평가 구간 1년(현재 모델의 테스트 연도)만 비교한 결과입니다. 규칙의 기준값은 "
            "이 구간을 보고 고르지 않았지만, 한 구간 결과로 일반화할 수 없습니다."
        ),
        "disclaimer": DISCLAIMER,
    }


@lru_cache(maxsize=1)
def get_guard_backtest() -> dict:
    """모델·태그가 바뀌지 않으므로 한 번만 계산한다 (재생 2회, 수 초)."""
    return build_guard_backtest(get_policy(), get_risk_tags())


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 첫 요청이 5초 제한을 넘지 않도록 시작할 때 미리 계산한다
    try:
        schedule = get_schedule()
        logger.info("모델 로드 완료: %s", schedule.model_name)
    except Exception:
        logger.exception("모델 로드 실패: /optimize는 503을 반환합니다")
    try:
        logger.info("리스크 태그 %d건 로드", len(get_risk_tags()))
    except Exception as error:
        logger.warning("리스크 태그 없음: Safe-Guard 없이 모델 비중만 응답 (%s)", error)
    try:
        result = get_guard_backtest()
        logger.info(
            "Safe-Guard 켬/끔 백테스트 완료 (공개 결과 대조: %s)",
            result["reproduces_published"],
        )
    except Exception as error:
        logger.warning("Safe-Guard 켬/끔 백테스트를 계산하지 못함 (%s)", error)
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
    risk_override: Optional[bool] = Field(
        default=True, description="리스크 태그 Safe-Guard 적용 여부"
    )


def _load_tags() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """(태그 표, 실패 사유). 태그가 없어도 서버는 모델 비중으로 계속 응답한다."""
    try:
        return get_risk_tags(), None
    except Exception as error:
        return None, f"리스크 태그를 읽을 수 없습니다: {error}"


@app.get("/health")
def health_check():
    tags, _ = _load_tags()
    base = {
        "service": "rl-engine",
        "version": app.version,
        "is_mock": False,
        "safeguard_mdd_limit": 0.15,
        "risk_tags_loaded": tags is not None,
        "risk_tag_count": 0 if tags is None else int(len(tags)),
        # 모델 출처: 파일 또는 MLflow 모델 저장소(버전·별칭)
        "model_source": model_source_info(),
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


def _round(values: pd.Series, digits: int = 6) -> List[Optional[float]]:
    return [_clean(round(float(v), digits)) if pd.notna(v) else None for v in values]


def build_learning_curves(frame: pd.DataFrame) -> dict:
    """실행별 학습 곡선과 요약 (시작·마지막 평균 보상, 마지막 에피소드 길이)."""
    runs = []
    for run, curve in frame.groupby("run", sort=True):
        curve = curve.sort_values("step")
        reward = curve["ep_rew_mean"]
        first = curve.iloc[0]
        runs.append(
            {
                "run": run,
                "stage": first["stage"],
                "window": int(first["window"]),
                "variant": first["variant"],
                "seed": int(first["seed"]),
                "steps": [int(v) for v in curve["step"]],
                "ep_rew_mean": _round(reward),
                "ep_len_mean": _round(curve["ep_len_mean"], 2),
                # 처음·마지막 5개 기록 평균 (한 점의 흔들림에 덜 민감하게)
                "start_reward": _clean(reward.head(5).mean()),
                "end_reward": _clean(reward.tail(5).mean()),
                "end_episode_length": _clean(curve["ep_len_mean"].tail(5).mean()),
                "total_steps": int(curve["step"].max()),
            }
        )
    return {
        "status": "ok",
        "metric": "rollout/ep_rew_mean",
        "current_model": model_source()["name"],
        "run_count": len(runs),
        "runs": runs,
        "note": (
            "Episode Reward는 보상 함수마다 식이 달라(단순 수익률·샤프비율·MDD 페널티) "
            "값의 크기를 서로 비교할 수 없습니다. 같은 보상 함수 안에서 수렴 여부를 봅니다."
        ),
        "source": "rl-engine 학습 TensorBoard 기록 (PPO, 100,352 스텝)",
        "disclaimer": DISCLAIMER,
    }


@lru_cache(maxsize=1)
def get_learning_curves() -> dict:
    return build_learning_curves(pd.read_csv(RL_LEARNING_CURVES))


@app.get("/learning-curves")
def learning_curves():
    """PPO 학습 곡선 (보상 함수 5종 × 구간 4 × seed 3, 최종·선택용 학습)."""
    try:
        return get_learning_curves()
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(
            503, f"학습 곡선 기록을 읽을 수 없습니다: {error}"
        ) from error


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

    def by_request_order(values: np.ndarray) -> Dict[str, float]:
        by_ticker = dict(zip(schedule.tickers, values))
        return {t: round(float(by_ticker[t]), 6) for t in req.tickers}

    raw = schedule.weights[index]
    requested = bool(req.risk_override)
    tags, reason = _load_tags()
    guard = {
        "requested": requested,
        "available": tags is not None,
        "rule": _guard_rule(schedule),
        "moved_weight": 0.0,
        "adjustments": [],
        "tags": [],
    }
    final = raw
    if tags is None:
        guard["reason"] = reason
    else:
        decision = guard_decision(schedule, index, tags)
        # 끈 상태에서도 어떤 신호가 있었는지는 보여 준다
        guard["tags"] = decision["tags"]
        if requested and decision["adjustments"]:
            final = decision["weights"]
            guard["adjustments"] = decision["adjustments"]
            guard["moved_weight"] = decision["moved_weight"]
    applied = bool(guard["adjustments"])

    message = (
        f"PPO 정책({schedule.model_name})의 목표 비중 "
        "(수수료 0.015%, 슬리피지 0.05% 명세값으로 학습·평가)"
    )
    if applied:
        message += (
            f" · 리스크 태그 Safe-Guard가 {len(guard['adjustments'])}개 자산 비중을 "
            f"{guard['moved_weight']:.1%}p 줄임"
        )

    return {
        "status": "ok",
        "service": "rl-engine",
        "is_mock": False,
        "model_used": schedule.model_name,
        "as_of": req.as_of,
        "decision_date": str(schedule.dates[index].date()),
        "weights": by_request_order(final),
        # Safe-Guard 적용 전 모델 원래 비중 (백테스트와 같은 값)
        "model_weights": by_request_order(raw),
        "transaction_cost_bps": 1.5,
        "slippage_bps": 5.0,
        # MDD 한도 Safe-Guard는 학습 환경의 조기 종료 규칙이라 서빙 응답에서는 항상 False
        "safeguard_triggered": False,
        "risk_override_applied": applied,
        "risk_guard": guard,
        "message": message,
        "disclaimer": DISCLAIMER,
    }


@app.get("/risk-tags")
def risk_tags():
    """리스크 태그 Safe-Guard가 현재 모델의 평가 구간에서 작동한 내역."""
    try:
        schedule = get_schedule()
    except Exception as error:
        raise HTTPException(503, f"모델을 사용할 수 없습니다: {error}") from error
    tags, reason = _load_tags()
    if tags is None:
        raise HTTPException(503, reason)
    return get_guard_summary(schedule, tags)


@app.get("/risk-tags/backtest")
def risk_tags_backtest():
    """리스크 태그 Safe-Guard를 끈 경우와 켠 경우의 평가 구간 성과 비교."""
    try:
        get_policy()
    except Exception as error:
        raise HTTPException(503, f"모델을 사용할 수 없습니다: {error}") from error
    tags, reason = _load_tags()
    if tags is None:
        raise HTTPException(503, reason)
    return get_guard_backtest()
