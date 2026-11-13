import time

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import gateway
import rl_server
from gateway import app as gateway_app
from rl_server import app as rl_app

gateway_client = TestClient(gateway_app)
rl_client = TestClient(rl_app)


# =====================================================================
# Part 1. RL Serving Server (8002) 고유 기능 유닛 테스트
#   CI에는 torch가 없으므로 모델 대신 고정된 비중 표를 넣어 API 동작을 검증한다
# =====================================================================

ALL_TICKERS = [
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


@pytest.fixture
def schedule(monkeypatch):
    """거래일 3일치 가짜 비중 표. i번째 날은 NVDA 비중이 0.1 * (i + 1)이다."""
    n = len(ALL_TICKERS)
    table = np.zeros((3, n))
    for day in range(3):
        nvda = 0.1 * (day + 1)
        table[day] = (1 - nvda) / (n - 1)
        table[day, ALL_TICKERS.index("NVDA")] = nvda
    fake = rl_server.PolicySchedule(
        model_name="fake_model",
        tickers=ALL_TICKERS,
        dates=pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"]),
        weights=table,
    )
    monkeypatch.setattr(rl_server, "get_schedule", lambda: fake)
    return fake


def _unavailable():
    raise FileNotFoundError("모델 파일이 없습니다")


def _optimize(**body):
    return rl_client.post("/optimize", json={"tickers": ALL_TICKERS, **body})


def test_01_rl_server_health(schedule):
    """1. /health가 모델 로드 상태와 선택 가능한 기준일을 반환하는지 검증"""
    data = rl_client.get("/health").json()
    assert data["status"] == "ok"
    assert data["service"] == "rl-engine"
    assert data["is_mock"] is False
    assert data["model_name"] == "fake_model"
    assert (data["as_of_min"], data["as_of_max"]) == ("2025-01-03", "2025-01-07")
    assert data["safeguard_mdd_limit"] == 0.15


def test_02_rl_optimize_weights_sum_to_one(schedule):
    """2. /optimize 비중 합이 1.0이고 요청한 자산을 모두 포함하는지 검증"""
    res = _optimize()
    assert res.status_code == 200
    weights = res.json()["weights"]
    assert set(weights) == set(ALL_TICKERS)
    assert abs(sum(weights.values()) - 1.0) < 1e-4


def test_03_rl_optimize_deterministic_reproducibility(schedule):
    """3. 동일 시점 요청 시 동일 비중을 반환하고 자산 순서에 영향받지 않는지 검증"""
    first = _optimize(as_of="2025-01-04T00:00:00-05:00").json()["weights"]
    res = rl_client.post(
        "/optimize",
        json={"tickers": ALL_TICKERS[::-1], "as_of": "2025-01-04T00:00:00-05:00"},
    )
    assert res.json()["weights"] == first


def test_04_rl_optimize_rejects_partial_tickers(schedule):
    """4. 모델이 지원하지 않는 자산 조합은 422로 거절하는지 검증"""
    for tickers in ([], ["SPY", "TLT"], ALL_TICKERS[:-1] + ["TSLA"]):
        assert rl_client.post("/optimize", json={"tickers": tickers}).status_code == 422


def test_05_rl_optimize_transaction_cost_metadata(schedule):
    """5. 거래비용(1.5bps)·슬리피지(5.0bps) 명세와 Safe-Guard 필드 반환 검증"""
    data = _optimize().json()
    assert data["transaction_cost_bps"] == 1.5
    assert data["slippage_bps"] == 5.0
    assert "safeguard_triggered" in data


@pytest.mark.parametrize(
    "as_of, decision_date, nvda",
    [
        (None, "2025-01-06", 0.3),
        # 01-03 0시에는 01-03 장이 끝나지 않았으므로 01-02 결정을 쓴다
        ("2025-01-03T00:00:00+00:00", "2025-01-02", 0.1),
        ("2025-01-04T00:00:00-05:00", "2025-01-03", 0.2),
        # 주말에는 직전 거래일(금요일) 결정을 쓴다
        ("2025-01-05", "2025-01-03", 0.2),
        ("2025-01-07", "2025-01-06", 0.3),
    ],
)
def test_rl_optimize_uses_only_past_information(schedule, as_of, decision_date, nvda):
    """기준 시각에 이미 끝난 거래일의 결정만 사용하는지(미래 정보 누수 방지) 검증"""
    data = _optimize(as_of=as_of).json()
    assert data["decision_date"] == decision_date
    assert data["weights"]["NVDA"] == pytest.approx(nvda)


@pytest.mark.parametrize("as_of", ["2025-01-02", "2025-01-08", "not-a-date"])
def test_rl_optimize_rejects_invalid_as_of(schedule, as_of):
    """평가 구간 밖이거나 형식이 잘못된 기준일은 422로 거절하는지 검증"""
    assert _optimize(as_of=as_of).status_code == 422


def test_rl_server_reports_unavailable_model(monkeypatch):
    """모델을 쓸 수 없으면 /optimize는 503, /health는 degraded를 반환하는지 검증"""
    monkeypatch.setattr(rl_server, "get_schedule", _unavailable)
    assert _optimize().status_code == 503
    data = rl_client.get("/health").json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


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


def test_08_gateway_optimize_response_and_sla(monkeypatch):
    """8. Gateway POST /optimize가 RL 응답을 그대로 전달하고 5초 이내에 응답하는지 검증"""
    upstream = {"status": "ok", "weights": {t: 1 / 12 for t in ALL_TICKERS}}
    _rl_upstream(monkeypatch, 200, upstream)
    start = time.perf_counter()
    res = gateway_client.post("/optimize", json={"tickers": ALL_TICKERS})
    elapsed = time.perf_counter() - start
    assert res.status_code == 200
    assert elapsed < 5.0, f"응답 시간 5초 초과: {elapsed:.2f}s"
    assert res.json() == upstream
    assert abs(sum(res.json()["weights"].values()) - 1.0) < 1e-6


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
        json={
            "question": "엔비디아 실적 발표",
            "tickers": ["NVDA"],
            "as_of": "2024-12-31T00:00:00-05:00",
        },
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
    assert all(
        s["status"] in ("offline", "error") for s in res.json()["services"].values()
    )


def test_14_rl_response_is_real_with_disclaimer(schedule):
    """14. 응답이 실제 모델 결과로 표시되고 금융 면책 문구가 포함되는지 검증"""
    data = _optimize().json()
    assert data["is_mock"] is False
    assert data["risk_override_applied"] is False
    assert "투자 조언" in data["disclaimer"]


def _rl_upstream(monkeypatch, status_code, payload):
    """Gateway가 호출하는 RL 서버 응답을 지정한 값으로 대체한다."""

    def handler(_):
        return httpx.Response(status_code, json=payload)

    original = httpx.AsyncClient

    def client(**kwargs):
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(gateway.httpx, "AsyncClient", client)


def test_gateway_passes_validation_error_from_rl(monkeypatch):
    """Gateway가 RL 서버의 422를 대체 비중으로 가리지 않고 전달하는지 검증"""
    _rl_upstream(monkeypatch, 422, {"detail": "이 모델은 12개 자산 전체만 지원합니다"})
    res = gateway_client.post("/optimize", json={"tickers": ["SPY"]})
    assert res.status_code == 422
    assert "12개" in res.json()["detail"]


def test_gateway_falls_back_when_rl_unavailable(monkeypatch):
    """RL 서버 장애(503) 시 Gateway가 mock_fallback으로 표시해 응답하는지 검증"""
    _rl_upstream(monkeypatch, 503, {"detail": "모델을 사용할 수 없습니다"})
    data = gateway_client.post("/optimize", json={"tickers": ["SPY", "TLT"]}).json()
    assert data["status"] == "mock_fallback"
    assert data["weights"] == {"SPY": 0.5, "TLT": 0.5}
