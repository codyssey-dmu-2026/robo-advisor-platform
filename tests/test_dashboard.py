"""대시보드·시계열 API 테스트.

- 시계열 API: 임시 산출물 폴더로 정상/없는 실행/잘못된 ID를 검증
- 대시보드: Gateway가 꺼져 있어도 모든 페이지가 예외 없이 안내 화면을 보여주는지 검증
"""

import os
from pathlib import Path

# 대시보드 모듈이 처음 import될 때 주소를 읽으므로, 연결되지 않는 주소를 먼저 지정
os.environ["GATEWAY_URL"] = "http://127.0.0.1:9"

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
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
