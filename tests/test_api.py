import time
import pytest
from fastapi.testclient import TestClient

from gateway import app as gateway_app
from rl_server import app as rl_app

gateway_client = TestClient(gateway_app)
rl_client = TestClient(rl_app)


# =====================================================================
# Part 1. RL Serving Server (8002) 고유 기능 유닛 테스트 (5개)
# =====================================================================

def test_01_rl_server_health():
    """1. RL 서빙 서버(/health) 정상 동작 및 필수 메타데이터 반환 검증"""
    res = rl_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "rl-engine"
    assert data["safeguard_mdd_limit"] == 0.15


def test_02_rl_optimize_weights_sum_to_one():
    """2. RL 최적화(/optimize) 결과 자산 비중의 합이 1.0(100%)인지 검증"""
    tickers = ["SPY", "QQQ", "TLT", "GLD", "NVDA", "005930"]
    res = rl_client.post("/optimize", json={"tickers": tickers, "as_of": "2024-12-31T00:00:00-05:00"})
    assert res.status_code == 200
    weights = res.json()["weights"]
    assert len(weights) == len(tickers)
    assert abs(sum(weights.values()) - 1.0) < 1e-2


def test_03_rl_optimize_deterministic_reproducibility():
    """3. 동일 시점(as_of) 및 동일 티커 요청 시 재현 가능한(Deterministic) 동일 비중 반환 검증"""
    payload = {"tickers": ["AAPL", "MSFT", "NVDA", "GLD"], "as_of": "2024-06-30T00:00:00-05:00"}
    res1 = rl_client.post("/optimize", json=payload)
    res2 = rl_client.post("/optimize", json=payload)
    assert res1.json()["weights"] == res2.json()["weights"]


def test_04_rl_optimize_empty_tickers_handling():
    """4. 빈 티커 리스트 입력 시 서버 크래시 없이 안전하게 처리하는지 검증"""
    res = rl_client.post("/optimize", json={"tickers": []})
    assert res.status_code == 200
    assert res.json()["weights"] == {}


def test_05_rl_optimize_transaction_cost_metadata():
    """5. 거래비용(0.015% = 1.5bps) 및 슬리피지(0.05% = 5.0bps) 명세 반영 여부 검증"""
    res = rl_client.post("/optimize", json={"tickers": ["SPY", "TLT"]})
    data = res.json()
    assert data["transaction_cost_bps"] == 1.5
    assert data["slippage_bps"] == 5.0
    assert "safeguard_triggered" in data


# =====================================================================
# Part 2. API Gateway (8080) 5대 엔드포인트 & 규격/SLA 테스트 (7개)
# =====================================================================

def test_06_gateway_openapi_five_endpoints_registered():
    """6. Swagger UI(OpenAPI) 스키마에 5대 필수 엔드포인트가 모두 등록되었는지 검증"""
    res = gateway_client.get("/openapi.json")
    assert res.status_code == 200
    paths = res.json().get("paths", {})
    required_endpoints = ["/health", "/optimize", "/explain", "/research", "/backtest"]
    for ep in required_endpoints:
        assert ep in paths, f"필수 엔드포인트 누락: {ep}"


def test_07_gateway_health_aggregation():
    """7. Gateway /health 엔드포인트가 3개 하위 마이크로서비스 상태를 통합 반환하는지 검증"""
    res = gateway_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["gateway"] == "ok"
    assert "evaluation_xai" in data["services"]
    assert "agentic_research" in data["services"]
    assert "rl_engine" in data["services"]


def test_08_gateway_optimize_response_and_sla():
    """8. Gateway POST /optimize 정상 응답 및 5초 이내 응답(SLA) 준수 검증"""
    start = time.perf_counter()
    res = gateway_client.post(
        "/optimize",
        json={"tickers": ["005930", "NVDA", "SPY", "TLT"], "risk_override": True}
    )
    elapsed = time.perf_counter() - start
    assert res.status_code == 200
    assert elapsed < 5.0, f"응답 시간 5초 초과: {elapsed:.2f}s"
    weights = res.json()["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-2


def test_09_gateway_explain_validation_error_422():
    """9. POST /explain 잘못된 요청(필수 필드 누락) 시 422 Validation Error 방어 검증"""
    res = gateway_client.post("/explain", json={"invalid_field": "wrong"})
    assert res.status_code == 422


def test_10_gateway_explain_valid_schema():
    """10. POST /explain 정상 스키마(run_id, decision_id, target_asset) 요청 처리 검증"""
    payload = {"run_id": "toy_policy", "decision_id": "toy0", "target_asset": "SYN00"}
    res = gateway_client.post("/explain", json=payload)
    # 하위 서버(8000) 구동 시 200, 미구동 환경(단독 테스트) 시 503으로 정상 라우팅 검증
    assert res.status_code in (200, 503)
    if res.status_code == 200:
        assert "predicted_weight" in res.json() or "shap_values" in res.json()


def test_11_gateway_research_validation_and_routing():
    """11. POST /research 필수 필드 검증 및 에이전트 리서치 라우팅 검증"""
    # 질문(question) 타입 오류 시 422 방어 확인
    bad_res = gateway_client.post("/research", json={"question": ["not_a_string"]})
    assert bad_res.status_code == 422

    # 정상 스키마 요청 시 5초 이내 응답 확인
    start = time.perf_counter()
    good_res = gateway_client.post(
        "/research",
        json={"question": "엔비디아 실적 발표", "tickers": ["NVDA"], "as_of": "2024-12-31T00:00:00-05:00"}
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert good_res.status_code in (200, 503)


def test_12_gateway_backtest_routing_and_sla():
    """12. GET /backtest 백테스트 성과 지표 라우팅 및 5초 이내 응답 검증"""
    start = time.perf_counter()
    res = gateway_client.get("/backtest")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert res.status_code in (200, 503)
    if res.status_code == 200:
        assert "runs" in res.json()

# =====================================================================
# Part 3. 통합 버그 회귀 테스트
# =====================================================================

def test_13_gateway_health_parallel_when_services_offline(monkeypatch):
    """13. 하위 서비스가 모두 응답하지 않아도 /health가 병렬 확인으로 3초 안에 응답하는지 검증"""
    import gateway
    # 연결되지 않는 주소(TEST-NET-1)로 바꿔 타임아웃을 유도
    for name in ("EVAL_API_URL", "AGENT_API_URL", "RL_API_URL"):
        monkeypatch.setattr(gateway, name, "http://192.0.2.1:9")
    monkeypatch.setattr(gateway, "HEALTH_TIMEOUT_S", 1.0)
    start = time.perf_counter()
    res = gateway_client.get("/health")
    elapsed = time.perf_counter() - start
    assert res.status_code == 200
    assert elapsed < 3.0, f"순차 호출 의심: {elapsed:.2f}s"
    assert all(s["status"] in ("offline", "error") for s in res.json()["services"].values())


def test_14_rl_mock_response_is_flagged_with_disclaimer():
    """14. 실제 PPO 연동 전 응답에 mock 표시와 금융 면책 문구가 포함되는지 검증"""
    data = rl_client.post("/optimize", json={"tickers": ["SPY", "TLT"]}).json()
    assert data["is_mock"] is True
    assert "투자 조언" in data["disclaimer"]
