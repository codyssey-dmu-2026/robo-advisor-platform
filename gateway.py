from fastapi import FastAPI, HTTPException, Path as PathParam
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from pathlib import Path
import asyncio
import math
import httpx
import os

app = FastAPI(
    title="통합 자율 로보어드바이저 API Gateway",
    version="1.0.0",
    description=(
        "5대 필수 엔드포인트(/health, /optimize, /explain, /research, /backtest) "
        "단일 진입점 서버"
    ),
)

EVAL_API_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000")
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8001")
RL_API_URL = os.getenv("RL_API_URL", "http://127.0.0.1:8002")

# 하위 서비스 호출 타임아웃(초). 요청 1건 5초 SLA 안에 Gateway 오버헤드를 남기도록 설정
HEALTH_TIMEOUT_S = float(os.getenv("GATEWAY_HEALTH_TIMEOUT_S", "2.0"))
UPSTREAM_TIMEOUT_S = float(os.getenv("GATEWAY_UPSTREAM_TIMEOUT_S", "4.5"))
RESEARCH_TIMEOUT_S = float(os.getenv("GATEWAY_RESEARCH_TIMEOUT_S", "4.8"))

# evaluation-xai 산출물 폴더 (읽기 전용). 일별 시계열(ledger/weights)은 evaluation-xai API에 없어서
# Gateway가 직접 읽는다. evaluation-xai에 같은 API가 생기면 프록시로 교체한다
EVAL_ARTIFACT_DIR = Path(
    os.getenv(
        "EVAL_ARTIFACT_DIR",
        Path(__file__).resolve().parent.parent
        / "evaluation-xai"
        / "artifacts"
        / "demo",
    )
)
DISCLAIMER = "본 결과는 교육 목적이며 실제 투자 조언에 사용할 수 없습니다. 백테스팅 성과는 미래 수익을 보장하지 않습니다."


# --- Pydantic 스키마 정의 (Swagger UI 자동 생성용) ---
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
        ],
        description="최적화 대상 12개 자산 티커",
    )
    as_of: Optional[str] = Field(default=None, description="기준 시각. 없으면 최신")
    risk_override: Optional[bool] = Field(default=True, description="RiskTag 연동 여부")


class ExplainRequest(BaseModel):
    run_id: str = Field(..., description="실행 ID")
    decision_id: str = Field(..., description="의사결정 ID")
    target_asset: str = Field(..., description="대상 자산 티커")


class ResearchRequest(BaseModel):
    question: str = Field(
        default="엔비디아 2024년 실적 발표", description="투자 리서치 질문"
    )
    tickers: Optional[List[str]] = Field(default=["NVDA"], description="관련 티커 목록")
    as_of: Optional[str] = Field(
        default="2024-12-31T00:00:00-05:00", description="기준 시각 (ISO 8601)"
    )


# 1. GET /health : 서버 상태 및 모델 로드 여부 반환
@app.get("/health", summary="1. 전체 서비스 상태 및 모델 로드 여부 확인")
async def health_check():
    targets = [
        ("evaluation_xai", EVAL_API_URL),
        ("agentic_research", AGENT_API_URL),
        ("rl_engine", RL_API_URL),
    ]

    async def probe(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
        try:
            r = await client.get(f"{url}/health")
            return (
                r.json()
                if r.status_code == 200
                else {"status": "error", "code": r.status_code}
            )
        except Exception:
            return {"status": "offline"}

    # 3개 서비스를 동시에 확인해 최악의 경우에도 HEALTH_TIMEOUT_S 근처에서 응답
    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
        results = await asyncio.gather(*(probe(client, url) for _, url in targets))
    return {
        "gateway": "ok",
        "services": {name: res for (name, _), res in zip(targets, results)},
    }


# 2. POST /optimize : 현재 시장 데이터 기반 최적 포트폴리오 비중 반환
@app.post("/optimize", summary="2. 강화학습 기반 최적 포트폴리오 비중 추론")
async def optimize_portfolio(req: OptimizeRequest):
    r = None
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
        try:
            r = await client.post(f"{RL_API_URL}/optimize", json=req.model_dump())
        except Exception:
            pass
    if r is not None and r.status_code == 200:
        return r.json()
    if r is not None and r.status_code == 422:
        # 잘못된 자산 조합·기준일은 대체 비중으로 가리지 않고 그대로 알린다
        raise HTTPException(status_code=422, detail=r.json().get("detail"))

    # RL 서버 장애 시 동일 비중으로 대체 (status로 구분 가능)
    n = len(req.tickers)
    w = round(1.0 / n, 4) if n > 0 else 0.0
    return {
        "status": "mock_fallback",
        "message": "RL Engine(8002)을 사용할 수 없어 동일 비중을 임시 반환",
        "as_of": req.as_of,
        "weights": {t: w for t in req.tickers},
        "safeguard_triggered": False,
        "disclaimer": "본 결과는 교육 목적이며 실제 투자 조언에 사용할 수 없습니다.",
    }


# 3. POST /explain : 특정 의사결정에 대한 SHAP 해석 반환
@app.post("/explain", summary="3. 특정 의사결정에 대한 SHAP 해석 반환")
async def explain_decision(req: ExplainRequest):
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
        try:
            r = await client.post(f"{EVAL_API_URL}/explain", json=payload)
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.json())
            return r.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Evaluation API 통신 실패: {e}"
            )


# 4. POST /research : 투자 질문에 대한 에이전틱 리서치 결과 반환
@app.post("/research", summary="4. 투자 질문에 대한 LangGraph 에이전틱 리서치 수행")
async def research_market(req: ResearchRequest):
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    async with httpx.AsyncClient(timeout=RESEARCH_TIMEOUT_S) as client:
        try:
            r = await client.post(f"{AGENT_API_URL}/research", json=payload)
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.json())
            return r.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Research API 통신 실패: {e}")


# 5. GET /backtest : 백테스트 성과 지표 반환
@app.get(
    "/backtest", summary="5. Walk-Forward 백테스트 12개 지표 및 ANOVA 검증 결과 반환"
)
async def get_backtest():
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
        try:
            r = await client.get(f"{EVAL_API_URL}/backtest")
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.json())
            return r.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Evaluation API 통신 실패: {e}"
            )


# 6. GET /backtest/{run_id}/timeseries : 대시보드 차트용 일별 시계열 (5대 필수 엔드포인트 외 보조 API)
def _clean(values) -> List[Optional[float]]:
    # NaN/inf는 JSON으로 보낼 수 없으므로 None으로 변환
    return [
        (
            None
            if v is None or (isinstance(v, float) and not math.isfinite(v))
            else float(v)
        )
        for v in values
    ]


@app.get(
    "/backtest/{run_id}/timeseries",
    summary="6. 백테스트 일별 자산가치·낙폭·비중 시계열 (대시보드 차트용)",
)
def get_backtest_timeseries(
    run_id: str = PathParam(
        ...,
        pattern=r"^[A-Za-z0-9_-]{1,80}$",
        description="백테스트 실행 ID (예: mvo_WF2023)",
    ),
):
    import pandas as pd  # 시계열 요청에서만 필요하므로 지연 로드

    run_dir = EVAL_ARTIFACT_DIR / run_id
    ledger_path, weights_path = run_dir / "ledger.parquet", run_dir / "weights.parquet"
    if not EVAL_ARTIFACT_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail="평가 산출물 폴더를 찾을 수 없습니다 (EVAL_ARTIFACT_DIR 확인)",
        )
    if not ledger_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"시계열이 없는 실행입니다: {run_id}"
        )

    ledger = pd.read_parquet(ledger_path)
    weights = (
        pd.read_parquet(weights_path)
        if weights_path.is_file()
        else pd.DataFrame(index=ledger.index)
    )
    equity = ledger["equity_end"]
    return {
        "run_id": run_id,
        "dates": [d.strftime("%Y-%m-%d") for d in pd.to_datetime(ledger.index)],
        "equity": _clean(equity),
        # 시작 자본 대비 누적 수익률 (차트 비교용으로 0에서 시작)
        "cumulative_return": _clean(equity / ledger["equity_start"].iloc[0] - 1.0),
        "drawdown": _clean(ledger["drawdown"]),
        "net_return": _clean(ledger["net_return_simple"]),
        "safeguard_triggered": [bool(v) for v in ledger["safeguard_triggered"]],
        "guard_active": [bool(v) for v in ledger["guard_active"]],
        "weights": {str(col): _clean(weights[col]) for col in weights.columns},
        "source": "evaluation-xai artifacts (read-only)",
        "disclaimer": DISCLAIMER,
    }
