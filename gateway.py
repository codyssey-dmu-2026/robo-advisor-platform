from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import httpx
import os

app = FastAPI(
    title="통합 자율 로보어드바이저 API Gateway",
    version="1.0.0",
    description="5대 필수 엔드포인트(/health, /optimize, /explain, /research, /backtest) 단일 진입점 서버"
)

EVAL_API_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000")
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8001")
RL_API_URL = os.getenv("RL_API_URL", "http://127.0.0.1:8002")

# 하위 서비스 호출 타임아웃(초). 요청 1건 5초 SLA 안에 Gateway 오버헤드를 남기도록 설정
HEALTH_TIMEOUT_S = float(os.getenv("GATEWAY_HEALTH_TIMEOUT_S", "2.0"))
UPSTREAM_TIMEOUT_S = float(os.getenv("GATEWAY_UPSTREAM_TIMEOUT_S", "4.5"))
RESEARCH_TIMEOUT_S = float(os.getenv("GATEWAY_RESEARCH_TIMEOUT_S", "4.8"))

# --- Pydantic 스키마 정의 (Swagger UI 자동 생성용) ---
class OptimizeRequest(BaseModel):
    tickers: List[str] = Field(
        default=["000660", "005930", "069500", "148070", "AAPL", "GLD", "MSFT", "NVDA", "QQQ", "SPY", "TLT", "VNQ"],
        description="최적화 대상 12개 자산 티커"
    )
    as_of: Optional[str] = Field(default="2024-12-31T00:00:00-05:00", description="기준 시각")
    risk_override: Optional[bool] = Field(default=True, description="RiskTag 연동 여부")

class ExplainRequest(BaseModel):
    run_id: str = Field(..., description="실행 ID")
    decision_id: str = Field(..., description="의사결정 ID")
    target_asset: str = Field(..., description="대상 자산 티커")

class ResearchRequest(BaseModel):
    question: str = Field(default="엔비디아 2024년 실적 발표", description="투자 리서치 질문")
    tickers: Optional[List[str]] = Field(default=["NVDA"], description="관련 티커 목록")
    as_of: Optional[str] = Field(default="2024-12-31T00:00:00-05:00", description="기준 시각 (ISO 8601)")


# 1. GET /health : 서버 상태 및 모델 로드 여부 반환
@app.get("/health", summary="1. 전체 서비스 상태 및 모델 로드 여부 확인")
async def health_check():
    targets = [("evaluation_xai", EVAL_API_URL), ("agentic_research", AGENT_API_URL), ("rl_engine", RL_API_URL)]

    async def probe(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
        try:
            r = await client.get(f"{url}/health")
            return r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
        except Exception:
            return {"status": "offline"}

    # 3개 서비스를 동시에 확인해 최악의 경우에도 HEALTH_TIMEOUT_S 근처에서 응답
    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
        results = await asyncio.gather(*(probe(client, url) for _, url in targets))
    return {"gateway": "ok", "services": {name: res for (name, _), res in zip(targets, results)}}


# 2. POST /optimize : 현재 시장 데이터 기반 최적 포트폴리오 비중 반환
@app.post("/optimize", summary="2. 강화학습 기반 최적 포트폴리오 비중 추론")
async def optimize_portfolio(req: OptimizeRequest):
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
        try:
            # 추후 RL 엔진(8002)이 켜지면 자동 연결
            r = await client.post(f"{RL_API_URL}/optimize", json=req.model_dump())
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    
    # RL 엔진 연동 전 Fallback 응답 (인터페이스 검증용)
    n = len(req.tickers)
    w = round(1.0 / n, 4) if n > 0 else 0.0
    return {
        "status": "mock_fallback",
        "message": "RL Engine(8002) 연결 전 기본 비중 반환",
        "as_of": req.as_of,
        "weights": {t: w for t in req.tickers},
        "safeguard_triggered": False,
        "disclaimer": "본 결과는 교육 목적이며 실제 투자 조언에 사용할 수 없습니다."
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
            raise HTTPException(status_code=503, detail=f"Evaluation API 통신 실패: {e}")


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
@app.get("/backtest", summary="5. Walk-Forward 백테스트 12개 지표 및 ANOVA 검증 결과 반환")
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
            raise HTTPException(status_code=503, detail=f"Evaluation API 통신 실패: {e}")