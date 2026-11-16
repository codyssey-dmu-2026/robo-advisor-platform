"""SHAP 설명 경로 테스트: 설명 전용 인스턴스 라우팅, 합성 데모 허용 전달, 목록·그림 프록시."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import gateway

client = TestClient(gateway.app)
REAL_ASYNC_CLIENT = httpx.AsyncClient

DEMO_BUNDLE = {
    "data_mode": "synthetic",
    "runs": {"toy_policy": {"metrics": {}}},
    "explanations": {
        "toy_policy/toy1": {"asset_names": ["SYN00", "SYN01"]},
        "toy_policy/toy0": {"asset_names": ["SYN00"]},
    },
    "plot_artifacts": {
        "toy_policy/toy0/SYN00": {"summary": "plots/a.png", "force": "plots/a.html"}
    },
}


@pytest.fixture
def upstream(monkeypatch):
    """하위 서비스 요청을 기록하고, 경로별로 정한 응답을 돌려준다."""
    calls = []
    routes = {}

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        for (host, path), response in routes.items():
            if request.url.host == host and request.url.path == path:
                return response(request)
        return httpx.Response(404, json={"detail": "없는 경로"})

    def make_client(**kwargs):
        return REAL_ASYNC_CLIENT(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(gateway, "EVAL_API_URL", "http://eval-main:8000")
    monkeypatch.setattr(gateway, "EVAL_EXPLAIN_API_URL", "http://eval-explain:8000")
    monkeypatch.setattr(gateway.httpx, "AsyncClient", make_client)
    return calls, routes


def test_explain_goes_to_explain_instance_with_synthetic_opt_in(upstream):
    """/explain은 설명 전용 인스턴스로 가고, allow_synthetic 값을 그대로 전달하는지 검증"""
    calls, routes = upstream
    routes[("eval-explain", "/explain")] = lambda r: httpx.Response(
        200,
        json={"run_id": "toy_policy", "decision_id": "toy0", "target_asset": "SYN00"},
    )
    body = {"run_id": "toy_policy", "decision_id": "toy0", "target_asset": "SYN00"}

    assert client.post("/explain", json=body).status_code == 200
    assert client.post("/explain", json=body | {"allow_synthetic": True}).is_success
    sent = [json.loads(c.content) for c in calls]
    assert [s["allow_synthetic"] for s in sent] == [False, True]
    assert {c.url.host for c in calls} == {"eval-explain"}


def test_explain_synthetic_refusal_is_passed_through(upstream):
    """평가 서비스가 합성 데모 설명을 409로 거절하면 그대로 409를 전달하는지 검증"""
    _, routes = upstream
    routes[("eval-explain", "/explain")] = lambda r: httpx.Response(
        409, json={"detail": "Opt in with allow_synthetic=true."}
    )
    body = {"run_id": "toy_policy", "decision_id": "toy0", "target_asset": "SYN00"}
    res = client.post("/explain", json=body)
    assert res.status_code == 409
    assert "allow_synthetic" in res.json()["detail"]


def test_explain_catalog_lists_decisions_assets_and_plots(upstream):
    """설명 목록이 설명 전용 인스턴스의 산출물에서 실행·결정별 자산과 그림 항목을 만드는지 검증"""
    _, routes = upstream
    routes[("eval-explain", "/backtest")] = lambda r: httpx.Response(
        200, json=DEMO_BUNDLE
    )
    data = client.get("/explain/catalog").json()
    assert data["data_mode"] == "synthetic"
    assert list(data["explanations"]) == ["toy_policy/toy0", "toy_policy/toy1"]
    assert data["explanations"]["toy_policy/toy1"] == ["SYN00", "SYN01"]
    assert data["plots"] == ["toy_policy/toy0/SYN00"]
    assert "투자 조언" in data["disclaimer"]


def test_explain_plots_are_proxied_with_type_and_isolation(upstream):
    """Summary(PNG)·Force(HTML) 그림을 형식 그대로 전달하고, HTML에는 격리 정책을 붙이는지 검증"""
    calls, routes = upstream
    base = "/explanations/toy_policy/toy0/SYN00/plots"
    routes[("eval-explain", f"{base}/summary")] = lambda r: httpx.Response(
        200, content=b"\x89PNG-bytes", headers={"content-type": "image/png"}
    )
    routes[("eval-explain", f"{base}/force")] = lambda r: httpx.Response(
        200,
        content=b"<html>force</html>",
        headers={"Content-Security-Policy": "sandbox allow-scripts"},
    )

    png = client.get(
        "/explain/plots/toy_policy/toy0/SYN00/summary?allow_synthetic=true"
    )
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content == b"\x89PNG-bytes"
    assert calls[-1].url.params["allow_synthetic"] == "true"

    html = client.get("/explain/plots/toy_policy/toy0/SYN00/force")
    assert html.headers["content-type"].startswith("text/html")
    assert html.headers["content-security-policy"].startswith("sandbox")
    assert calls[-1].url.params["allow_synthetic"] == "false"


def test_explain_plots_reject_bad_kind_and_path(upstream):
    """그림 종류·경로 값이 규칙에 맞지 않으면 하위 서비스에 보내기 전에 422로 거절하는지 검증"""
    calls, _ = upstream
    assert client.get("/explain/plots/toy_policy/toy0/SYN00/other").status_code == 422
    assert client.get("/explain/plots/bad.id/toy0/SYN00/summary").status_code == 422
    assert calls == []


def test_explain_plot_missing_is_404(upstream):
    """그림이 없는 결정이면 하위 서비스의 404를 그대로 전달하는지 검증"""
    res = client.get("/explain/plots/toy_policy/toy9/SYN00/summary")
    assert res.status_code == 404


def test_backtest_still_goes_to_main_instance(upstream):
    """/backtest(성과·ANOVA)는 설명 전용이 아닌 기본 평가 인스턴스로 가는지 검증"""
    calls, routes = upstream
    routes[("eval-main", "/backtest")] = lambda r: httpx.Response(
        200, json={"data_mode": "imported_unverified", "runs": {}}
    )
    assert client.get("/backtest").json()["data_mode"] == "imported_unverified"
    assert calls[-1].url.host == "eval-main"
