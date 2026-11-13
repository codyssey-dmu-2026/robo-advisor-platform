"""RL 서빙 서버 (Port 8002): rl-engine에서 학습한 PPO 모델로 자산 비중을 계산한다.

서버가 켜질 때 평가 구간 전체를 모델로 한 번 재생해 날짜별 목표 비중을 메모리에 올리고,
요청이 오면 그 표에서 찾아 돌려준다. rl-engine의 환경 코드를 그대로 쓰므로
백테스트(eval_test_daily.csv)와 같은 비중이 나온다.
모델·데이터는 환경 변수로 지정하므로 RL_MODEL_NAME만 바꾸면 모델을 교체할 수 있다.
"""

import logging
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
