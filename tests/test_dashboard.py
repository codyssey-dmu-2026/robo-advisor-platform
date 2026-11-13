"""대시보드·시계열 API 테스트.

- 시계열 API: 임시 산출물 폴더로 정상/없는 실행/잘못된 ID를 검증
- 대시보드: Gateway가 꺼져 있어도 모든 페이지가 예외 없이 안내 화면을 보여주는지 검증
"""

import os
from pathlib import Path

# 대시보드 모듈이 처음 import될 때 주소를 읽으므로, 연결되지 않는 주소를 먼저 지정
os.environ["GATEWAY_URL"] = "http://127.0.0.1:9"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from conftest import risk_tag  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import gateway  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
PAGES = [
    "views/portfolio.py",
    "views/risk.py",
    "views/performance.py",
    "views/shap.py",
    "views/anova.py",
    "views/research.py",
]


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, freq="B", name="date")
    run = tmp_path / "mvo_WF2024"
    run.mkdir()
    pd.DataFrame(
        {
            "equity_start": [100.0, 101.0, 99.0],
            "equity_end": [101.0, 99.0, 100.0],
            "net_return_simple": [0.01, -0.0198, 0.0101],
            "drawdown": [0.0, 0.0198, 0.0099],
            "safeguard_triggered": [False, False, False],
            "guard_active": [False, False, False],
        },
        index=dates,
    ).to_parquet(run / "ledger.parquet")
    pd.DataFrame(
        {"SPY": [0.6, 0.5, 0.5], "CASH": [0.4, 0.5, 0.5]}, index=dates
    ).to_parquet(run / "weights.parquet")
    monkeypatch.setattr(gateway, "EVAL_ARTIFACT_DIR", tmp_path)
    return tmp_path


def test_15_timeseries_returns_aligned_series(artifact_dir):
    """15. 시계열 API가 날짜·누적수익률·낙폭·비중을 같은 길이로 반환하는지 검증"""
    data = TestClient(gateway.app).get("/backtest/mvo_WF2024/timeseries").json()
    assert data["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert data["cumulative_return"][0] == pytest.approx(0.01)
    assert len(data["drawdown"]) == len(data["dates"]) == len(data["weights"]["SPY"])
    assert "투자 조언" in data["disclaimer"]


def test_16_timeseries_unknown_and_invalid_run(artifact_dir):
    """16. 없는 실행은 404, 경로 조작이 가능한 ID는 422로 거부하는지 검증"""
    client = TestClient(gateway.app)
    assert client.get("/backtest/nope/timeseries").status_code == 404
    assert client.get("/backtest/bad.id/timeseries").status_code == 422


class _Response:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def online_gateway(perf_results, write_tags, monkeypatch):
    """Gateway 응답을 실제 형식의 성과 데이터로 대체 (RL 성과 화면 렌더링 검증용)."""
    import requests
    import streamlit as st

    import rl_server

    monkeypatch.setattr(
        rl_server, "get_schedule", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    perf = rl_server.get_performance() | {"weights": None}
    tickers = [
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
    # 리스크 태그 Safe-Guard: 01-03 장 마감에 NVDA 위험 신호(점수 0.8)가 공개된 상황
    schedule = rl_server.PolicySchedule(
        model_name="fake",
        tickers=tickers,
        dates=pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"]),
        weights=np.full((3, len(tickers)), 1 / len(tickers)),
    )
    write_tags(risk_tag("NVDA", 0.8, "2025-01-03T16:00:00-05:00"))
    tags = rl_server.get_risk_tags()
    decision = rl_server.guard_decision(schedule, 2, tags)
    responses = {
        "/performance": perf,
        "/risk-tags": rl_server.build_guard_summary(schedule, tags),
        "/health": {
            "gateway": "ok",
            "services": {
                "rl_engine": {
                    "status": "ok",
                    "as_of_min": "2025-01-03",
                    "as_of_max": "2025-01-07",
                }
            },
        },
        "/optimize": {
            "status": "ok",
            "weights": dict(zip(tickers, decision["weights"].tolist())),
            "model_weights": dict(zip(tickers, schedule.weights[2].tolist())),
            "decision_date": "2025-01-06",
            "model_used": "fake",
            "risk_override_applied": True,
            "risk_guard": {
                "requested": True,
                "available": True,
                "rule": rl_server._guard_rule(schedule),
                "moved_weight": decision["moved_weight"],
                "adjustments": decision["adjustments"],
                "tags": decision["tags"],
            },
        },
    }

    def fake_request(method, url, **kwargs):
        path = url.removeprefix(os.environ["GATEWAY_URL"])
        if path not in responses:
            raise requests.exceptions.ConnectionError(path)
        return _Response(responses[path])

    monkeypatch.setattr(requests, "request", fake_request)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.mark.parametrize(
    "page", [None, "views/portfolio.py", "views/risk.py", "views/performance.py"]
)
def test_18_dashboard_renders_real_performance(online_gateway, page):
    """18. 실제 성과 데이터로 개요·포트폴리오·리스크·강화학습 성과 화면이 예외 없이 그려지는지 검증"""
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    texts = [m.value for m in at.markdown]
    # st.badge는 markdown으로 그려진다. 실데이터 표시가 있고 API 오류 카드는 없어야 함
    assert any("실데이터" in t for t in texts)
    failures = ("연결할 수 없습니다", "요청이 실패했습니다", "응답 시간이 초과")
    assert not any(f in t for t in texts for f in failures), "API 오류 카드가 표시됨"


def _page(path: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.switch_page(path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_19_portfolio_shows_risk_tag_adjustments(online_gateway):
    """19. 포트폴리오 화면이 리스크 태그 Safe-Guard의 조정 내역과 발동일 바로가기를 보여 주는지 검증"""
    at = _page("views/portfolio.py")
    texts = [m.value for m in at.markdown]
    assert any("리스크 태그 Safe-Guard 조정 내역" in t for t in texts)
    guard_kpi = next(m for m in at.metric if m.label == "리스크 태그 Safe-Guard")
    assert guard_kpi.value == "1개 축소"
    table = next(d.value for d in at.dataframe if "조정 후" in d.value.columns)
    assert table["조정 후"].iloc[0] == pytest.approx(1 / 12 * (1 - 0.5 * 0.8))
    assert at.toggle[0].value is True  # 기본값: 리스크 태그 반영 켜짐

    # 발동일 바로가기를 고르면 기준일이 그 날짜로 바뀐다
    jump = at.selectbox(key="pf_jump")
    assert list(jump.options) and jump.value is None
    jump.select("2025-01-04").run()
    assert not at.exception, [e.value for e in at.exception]
    assert str(at.date_input(key="pf_as_of").value) == "2025-01-04"


def test_20_risk_page_shows_risk_tag_guard_summary(online_gateway):
    """20. 리스크 화면이 리스크 태그 Safe-Guard 발동 현황(발동일 수)과 한계 안내를 보여 주는지 검증"""
    at = _page("views/risk.py")
    days = next(m for m in at.metric if m.label == "발동일")
    assert days.value == "2일"
    assert any("백테스트 성과에는" in c.value for c in at.caption)


@pytest.mark.parametrize("page", [None, *PAGES])
def test_17_dashboard_pages_render_when_gateway_offline(page):
    """17. Gateway 오프라인에서도 각 페이지가 예외 없이 렌더링되고 사이드바에 오프라인 안내가 뜨는지 검증"""
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.error, "사이드바에 Gateway 오프라인 안내가 없음"
