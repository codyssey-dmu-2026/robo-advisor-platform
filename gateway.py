from fastapi import FastAPI, HTTPException, Path as PathParam
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
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
    risk_override: Optional[bool] = Field(
        default=True,
        description="리스크 태그 Safe-Guard 적용 여부 (위험 신호가 강한 자산 비중 축소)",
    )


class ExplainRequest(BaseModel):
    # evaluation-xai /explain 요청 규칙과 같은 형식 제한 (잘못된 값은 Gateway에서 422)
    run_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,80}$", description="실행 ID")
    decision_id: str = Field(
        ..., pattern=r"^[A-Za-z0-9_-]{1,80}$", description="의사결정 ID"
    )
    target_asset: str = Field(
        ..., pattern=r"^[A-Za-z0-9_.-]{1,30}$", description="대상 자산 티커"
    )


class ResearchRequest(BaseModel):
    # agentic-research /research 요청 규칙과 동일. 종목·기준일은 강제하지 않는다
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "삼성전자 최근 실적과 리스크는?"},
                {
                    "question": "엔비디아 실적 발표가 주가에 준 영향은?",
                    "tickers": ["NVDA"],
                    "as_of": "2025-12-31T16:00:00-05:00",
                },
            ]
        }
    )

    question: str = Field(
        ..., min_length=2, max_length=500, description="투자 리서치 질문"
    )
    tickers: Optional[List[str]] = Field(
        default=None,
        max_length=12,
        description="관련 티커. 생략하면 에이전트가 질문에서 종목을 인식",
    )
    as_of: Optional[AwareDatetime] = Field(
        default=None,
        description="기준 시각(시간대 포함 ISO 8601). 이 시각 이전 문서만 사용, 생략하면 요청 시각",
    )


# --- 응답 스키마 (Swagger 문서용) ---
# 하위 서비스 응답을 그대로 전달하는 엔드포인트는 핵심 필드만 정의하고 나머지 필드도 통과시킨다
class PassThrough(BaseModel):
    model_config = ConfigDict(extra="allow")


class ServiceStatus(PassThrough):
    status: str = Field(description="ok / degraded / error / offline")


class HealthResponse(BaseModel):
    gateway: str
    services: Dict[str, ServiceStatus] = Field(
        description="evaluation_xai, agentic_research, rl_engine 상태"
    )


class OptimizeResponse(PassThrough):
    status: str = Field(
        description="ok: 실제 모델 / mock_fallback: RL 장애 시 동일 비중"
    )
    weights: Dict[str, float] = Field(
        description="자산별 최종 비중 (합 1.0, 리스크 태그 Safe-Guard 적용 후)"
    )
    model_weights: Optional[Dict[str, float]] = Field(
        default=None, description="Safe-Guard 적용 전 모델 원래 비중"
    )
    as_of: Optional[str] = None
    decision_date: Optional[str] = Field(
        default=None, description="비중을 결정한 거래일 (장 마감 후)"
    )
    model_used: Optional[str] = None
    is_mock: Optional[bool] = None
    safeguard_triggered: bool
    risk_override_applied: Optional[bool] = Field(
        default=None, description="리스크 태그로 비중이 실제로 조정됐는지"
    )
    risk_guard: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Safe-Guard 규칙, 근거 태그(tags), 자산별 조정 내역(adjustments)",
    )
    message: Optional[str] = None
    disclaimer: str


class RiskTagsResponse(PassThrough):
    model_name: str
    rule: Dict[str, Any] = Field(description="위험 점수 정의, 기준값, 최대 축소 비율")
    tag_count: int
    period: Dict[str, str] = Field(description="현재 모델의 평가 구간")
    trading_days: int
    adjusted_days: int = Field(description="Safe-Guard가 비중을 조정한 거래일 수")
    days: List[Dict[str, Any]] = Field(description="발동일별 줄어든 비중과 대상 자산")
    by_asset: List[Dict[str, Any]]
    tags: List[Dict[str, Any]] = Field(description="구간 안에서 유효했던 리스크 태그")
    limitation: str
    disclaimer: str


class ExplainResponse(PassThrough):
    run_id: str
    decision_id: str
    target_asset: str
    data_mode: Optional[str] = Field(default=None, description="synthetic / real")
    method: Optional[str] = None
    base_value: Optional[float] = None
    predicted_weight: Optional[float] = None
    groups: Optional[List[Any]] = Field(default=None, description="피처 그룹 이름")
    contributions: Optional[List[Any]] = Field(
        default=None, description="그룹별 SHAP 기여도"
    )
    interpretation: Optional[str] = None
    disclaimer: Optional[str] = None


class ResearchResponse(PassThrough):
    question: str
    answer: str
    opinion: Optional[str] = None
    confidence: Optional[float] = None
    claims: Optional[List[Dict[str, Any]]] = None
    citations: List[Dict[str, Any]] = Field(description="원문 출처")
    risk_tags: Optional[List[Any]] = None
    trace: List[Dict[str, Any]] = Field(description="단계별 추론 과정")
    elapsed_ms: Optional[int] = None
    disclaimer: Optional[str] = None


class BacktestResponse(PassThrough):
    data_mode: Optional[str] = Field(default=None, description="synthetic / real")
    runs: Dict[str, Dict[str, Any]] = Field(description="실행별 12개 지표")
    mission_anova: Optional[Any] = Field(default=None, description="ANOVA 검증 결과")
    bootstrap: Optional[Any] = None
    disclaimer: Optional[str] = None


class PerformanceResponse(PassThrough):
    data_mode: str
    model_name: str
    strategy: str
    seed: str
    benchmark: str
    windows: List[Dict[str, Any]] = Field(description="Walk-Forward 테스트 구간")
    strategies: List[Dict[str, Any]] = Field(
        description="현재 모델·동일가중·벤치마크·다른 보상 함수의 12개 지표"
    )
    series: Dict[str, Any] = Field(description="4년 연결 누적 수익률·낙폭")
    breaches: List[Dict[str, Any]] = Field(description="Safe-Guard 한도 초과 구간")
    mdd_limit: float
    weights: Optional[Dict[str, Any]] = Field(
        default=None, description="현재 모델의 날짜별 목표 비중"
    )
    disclaimer: str


class TimeseriesResponse(BaseModel):
    run_id: str
    dates: List[str]
    equity: List[Optional[float]]
    cumulative_return: List[Optional[float]]
    drawdown: List[Optional[float]]
    net_return: List[Optional[float]]
    safeguard_triggered: List[bool]
    guard_active: List[bool]
    weights: Dict[str, List[Optional[float]]]
    source: str
    disclaimer: str


async def _forward(
    method: str, url: str, service: str, *, timeout: float, json: Any = None
) -> Any:
    """하위 서비스 호출. 시간 초과는 504, 연결 실패는 503, 오류 응답은 상태 코드 그대로."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.request(method, url, json=json)
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=504,
                detail=f"{service} 응답이 {timeout:.1f}초 안에 오지 않았습니다 "
                "(요청 1건 5초 기준)",
            ) from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=503, detail=f"{service}에 연결할 수 없습니다: {e}"
            ) from e
    if r.status_code != 200:
        try:
            body = r.json()
            detail = body.get("detail", body) if isinstance(body, dict) else body
        except ValueError:
            detail = r.text[:200]
        raise HTTPException(status_code=r.status_code, detail=detail)
    return r.json()


# 1. GET /health : 서버 상태 및 모델 로드 여부 반환
@app.get(
    "/health",
    summary="1. 전체 서비스 상태 및 모델 로드 여부 확인",
    response_model=HealthResponse,
    response_model_exclude_unset=True,
)
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
@app.post(
    "/optimize",
    summary="2. 강화학습 기반 최적 포트폴리오 비중 추론",
    response_model=OptimizeResponse,
    response_model_exclude_unset=True,
)
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
        "disclaimer": DISCLAIMER,
    }


@app.get(
    "/performance",
    summary="(보조) 현재 RL 모델의 Walk-Forward 백테스트 성과",
    response_model=PerformanceResponse,
    response_model_exclude_unset=True,
)
async def rl_performance():
    """대시보드 성과·리스크 화면용. 동일가중·SPY·다른 보상 함수와의 비교 포함."""
    return await _forward(
        "GET",
        f"{RL_API_URL}/performance",
        "RL Engine(8002)",
        timeout=UPSTREAM_TIMEOUT_S,
    )


@app.get(
    "/risk-tags",
    summary="(보조) 리스크 태그 Safe-Guard 발동 내역",
    response_model=RiskTagsResponse,
    response_model_exclude_unset=True,
)
async def risk_tag_guard():
    """에이전트의 리스크 태그가 현재 모델 비중을 언제, 얼마나 줄였는지 요약."""
    return await _forward(
        "GET",
        f"{RL_API_URL}/risk-tags",
        "RL Engine(8002)",
        timeout=UPSTREAM_TIMEOUT_S,
    )


# 3. POST /explain : 특정 의사결정에 대한 SHAP 해석 반환
@app.post(
    "/explain",
    summary="3. 특정 의사결정에 대한 SHAP 해석 반환",
    response_model=ExplainResponse,
    response_model_exclude_unset=True,
)
async def explain_decision(req: ExplainRequest):
    return await _forward(
        "POST",
        f"{EVAL_API_URL}/explain",
        "Evaluation API(8000)",
        timeout=UPSTREAM_TIMEOUT_S,
        json=req.model_dump(),
    )


# 4. POST /research : 투자 질문에 대한 에이전틱 리서치 결과 반환
@app.post(
    "/research",
    summary="4. 투자 질문에 대한 LangGraph 에이전틱 리서치 수행",
    response_model=ResearchResponse,
    response_model_exclude_unset=True,
)
async def research_market(req: ResearchRequest):
    # 생략한 값은 보내지 않아 에이전트의 기본 동작(질문에서 종목 인식, 요청 시각 기준)을 따른다
    payload = req.model_dump(mode="json", exclude_none=True)
    return await _forward(
        "POST",
        f"{AGENT_API_URL}/research",
        "Research API(8001)",
        timeout=RESEARCH_TIMEOUT_S,
        json=payload,
    )


# 5. GET /backtest : 백테스트 성과 지표 반환
@app.get(
    "/backtest",
    summary="5. Walk-Forward 백테스트 12개 지표 및 ANOVA 검증 결과 반환",
    response_model=BacktestResponse,
    response_model_exclude_unset=True,
)
async def get_backtest():
    return await _forward(
        "GET",
        f"{EVAL_API_URL}/backtest",
        "Evaluation API(8000)",
        timeout=UPSTREAM_TIMEOUT_S,
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
    response_model=TimeseriesResponse,
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
