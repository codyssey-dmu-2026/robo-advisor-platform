"""리스크 태그 Safe-Guard 테스트.

에이전트가 만든 위험 이벤트(RiskTag)가 비중을 규칙대로 줄이는지, 미래 태그를 미리 쓰지
않는지, 태그 파일이 없어도 서버가 모델 비중으로 응답하는지 검증한다.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import gateway
import rl_server
from conftest import risk_tag

rl_client = TestClient(rl_server.app)
TICKERS = [
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
# NVDA 태그: 01-03 미국 장 마감(16:00 ET)에 공개, 5일간 유효, 위험 점수 0.8
NVDA_TAG = risk_tag("NVDA", 0.8, "2025-01-03T16:00:00-05:00")


@pytest.fixture
def schedule(monkeypatch):
    """거래일 3일치 가짜 비중 표. i번째 날은 NVDA 비중이 0.1 * (i + 1)이다."""
    n = len(TICKERS)
    table = np.zeros((3, n))
    for day in range(3):
        nvda = 0.1 * (day + 1)
        table[day] = (1 - nvda) / (n - 1)
        table[day, TICKERS.index("NVDA")] = nvda
    fake = rl_server.PolicySchedule(
        model_name="fake_model",
        tickers=TICKERS,
        dates=pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"]),
        weights=table,
    )
    monkeypatch.setattr(rl_server, "get_schedule", lambda: fake)
    return fake


def _optimize(**body) -> dict:
    res = rl_client.post("/optimize", json={"tickers": TICKERS, **body})
    assert res.status_code == 200, res.text
    return res.json()


# ---------------- 규칙 (순수 계산) ----------------
def test_cut_is_proportional_to_score_and_goes_to_untagged_assets():
    """점수 0.8이면 비중의 40%를 줄이고, 줄인 만큼을 나머지 자산이 원래 비율대로 받는지 검증"""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    out = rl_server.apply_risk_tags(weights, np.array([0.8, 0, 0, 0]), 0.4)
    assert out[0] == pytest.approx(0.24)
    assert out[1:] == pytest.approx([0.38, 0.2 + 0.16 / 3, 0.1 + 0.16 / 6])
    assert out.sum() == pytest.approx(1.0)


def test_weak_signal_leaves_weights_unchanged():
    """위험 점수가 기준(0.5) 미만이면 비중을 건드리지 않는지 검증"""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    out = rl_server.apply_risk_tags(weights, np.array([0.49, 0, 0.3, 0]), 0.4)
    assert out == pytest.approx(weights)


def test_redistribution_respects_weight_cap():
    """줄인 비중을 받는 자산도 자산당 상한(40%)을 넘지 않는지 검증"""
    weights = np.array([0.4, 0.38, 0.12, 0.10])
    out = rl_server.apply_risk_tags(weights, np.array([0.8, 0, 0, 0]), 0.4)
    assert out[0] == pytest.approx(0.24)
    assert out.max() <= 0.4 + 1e-9
    assert out.sum() == pytest.approx(1.0)


def test_all_assets_tagged_still_sums_to_one():
    """모든 자산에 같은 위험 신호가 있으면 옮길 곳이 없어 비중 합 1.0을 유지하는지 검증"""
    weights = np.full(12, 1 / 12)
    out = rl_server.apply_risk_tags(weights, np.full(12, 0.8), 0.4)
    assert out.sum() == pytest.approx(1.0)
    assert out == pytest.approx(weights)


def test_stronger_signal_is_cut_more_when_all_assets_tagged():
    """모든 자산이 태그돼도 신호가 더 강한 자산이 상대적으로 더 줄어드는지 검증"""
    out = rl_server.apply_risk_tags(
        np.full(4, 0.25), np.array([0.9, 0.5, 0.5, 0.5]), 0.4
    )
    assert out[0] < 0.25 < out[1]
    assert out.sum() == pytest.approx(1.0)


# ---------------- /optimize 연동 ----------------
def test_optimize_reduces_tagged_asset_and_explains_it(schedule, write_tags):
    """유효한 태그가 있으면 해당 자산 비중이 줄고, 근거 태그와 조정 내역을 함께 주는지 검증"""
    write_tags(NVDA_TAG)
    data = _optimize(as_of="2025-01-04")
    assert data["decision_date"] == "2025-01-03"
    assert data["model_weights"]["NVDA"] == pytest.approx(0.2)
    assert data["weights"]["NVDA"] == pytest.approx(0.12)  # 0.2 × (1 − 0.5 × 0.8)
    assert sum(data["weights"].values()) == pytest.approx(1.0, abs=1e-5)
    assert data["risk_override_applied"] is True
    guard = data["risk_guard"]
    assert guard["moved_weight"] == pytest.approx(0.08)
    assert guard["adjustments"] == [
        {"asset": "NVDA", "score": 0.8, "before": 0.2, "after": 0.12}
    ]
    assert guard["tags"][0]["source_url"] == "https://example.com/NVDA"
    assert "Safe-Guard" in data["message"]


def test_optimize_never_uses_a_tag_published_after_the_decision(schedule, write_tags):
    """결정일 장 마감 뒤에 공개된 태그는 쓰지 않는지(미래 정보 누수 방지) 검증"""
    write_tags(NVDA_TAG)
    data = _optimize(as_of="2025-01-03")  # 01-02 결정. 태그는 01-03 장 마감에 공개
    assert data["decision_date"] == "2025-01-02"
    assert data["weights"] == data["model_weights"]
    assert data["risk_override_applied"] is False
    assert data["risk_guard"]["tags"] == []


def test_optimize_ignores_expired_tag(schedule, write_tags):
    """만료된 태그는 더 이상 비중을 줄이지 않는지 검증"""
    write_tags(risk_tag("NVDA", 0.8, "2025-01-02T16:00:00-05:00", days=1))
    assert _optimize(as_of="2025-01-03")["risk_override_applied"] is True
    assert _optimize(as_of="2025-01-04")["risk_override_applied"] is False


def test_optimize_keeps_model_weights_when_guard_is_off(schedule, write_tags):
    """risk_override=false면 모델 비중 그대로 주되, 있었던 위험 신호는 알려 주는지 검증"""
    write_tags(NVDA_TAG)
    data = _optimize(as_of="2025-01-04", risk_override=False)
    assert data["weights"] == data["model_weights"]
    assert data["risk_override_applied"] is False
    assert data["risk_guard"]["requested"] is False
    assert [t["asset_id"] for t in data["risk_guard"]["tags"]] == ["NVDA"]


def test_optimize_lists_weak_tag_without_applying_it(schedule, write_tags):
    """기준 미만 신호는 목록에는 보이지만 비중에는 반영하지 않는지 검증"""
    write_tags(risk_tag("NVDA", 0.5, "2025-01-03T16:00:00-05:00", confidence=0.9))
    data = _optimize(as_of="2025-01-04")
    assert data["risk_override_applied"] is False
    assert data["risk_guard"]["tags"][0]["applied"] is False


def test_optimize_uses_strongest_tag_and_keeps_korean_ticker(schedule, write_tags):
    """자산의 위험 점수는 유효한 태그 중 최댓값이고, 0으로 시작하는 티커도 그대로 읽는지 검증"""
    write_tags(
        risk_tag("005930", 0.6, "2025-01-03T15:30:00+09:00"),
        risk_tag(
            "005930", 0.9, "2025-01-03T15:30:00+09:00", risk_type="earnings_shock"
        ),
    )
    data = _optimize(as_of="2025-01-04")
    (item,) = data["risk_guard"]["adjustments"]
    assert (item["asset"], item["score"]) == ("005930", 0.9)
    assert item["after"] == pytest.approx(item["before"] * (1 - 0.5 * 0.9), abs=1e-6)


def test_optimize_works_without_risk_tag_file(schedule):
    """태그 파일이 없어도 모델 비중으로 정상 응답하고 사유를 알려 주는지 검증"""
    data = _optimize()
    assert data["weights"] == data["model_weights"]
    assert data["risk_guard"]["available"] is False
    assert "리스크 태그" in data["risk_guard"]["reason"]
    health = rl_client.get("/health").json()
    assert (health["status"], health["risk_tags_loaded"]) == ("ok", False)


# ---------------- GET /risk-tags (발동 내역) ----------------
def test_risk_tags_summary_counts_adjusted_days(schedule, write_tags):
    """평가 구간에서 Safe-Guard가 작동한 날, 대상 자산, 근거 태그를 요약하는지 검증"""
    write_tags(NVDA_TAG, risk_tag("TLT", 0.9, "2024-06-03T16:00:00-04:00"))
    data = rl_client.get("/risk-tags").json()
    assert (data["trading_days"], data["adjusted_days"]) == (3, 2)
    assert [d["date"] for d in data["days"]] == ["2025-01-03", "2025-01-06"]
    assert data["days"][0]["as_of"] == "2025-01-04"
    assert data["days"][1]["moved_weight"] == pytest.approx(0.12)  # 0.3 × 0.4
    assert data["max_moved"] == pytest.approx(0.12)
    assert data["by_asset"] == [{"asset": "NVDA", "days": 2, "moved_weight": 0.1}]
    # 구간 밖(2024년) 태그는 전체 건수에는 들어가지만 구간 내 목록에는 없다
    assert data["tag_count"] == 2
    assert [t["asset_id"] for t in data["tags"]] == ["NVDA"]
    assert data["rule"]["threshold"] == 0.5
    assert "백테스트" in data["limitation"]
    assert rl_client.get("/health").json()["risk_tag_count"] == 2


def test_risk_tags_unavailable_returns_503(schedule, monkeypatch):
    """태그 파일이나 모델이 없으면 503으로 사유를 알리는지 검증"""
    res = rl_client.get("/risk-tags")
    assert res.status_code == 503
    assert "리스크 태그" in res.json()["detail"]

    def no_model():
        raise FileNotFoundError("모델 파일이 없습니다")

    monkeypatch.setattr(rl_server, "get_schedule", no_model)
    assert rl_client.get("/risk-tags").status_code == 503


def test_gateway_forwards_risk_tags(schedule, write_tags, monkeypatch):
    """Gateway GET /risk-tags가 RL 서버 응답을 그대로 전달하고, 장애 시 503인지 검증"""
    import httpx

    write_tags(NVDA_TAG)
    upstream = rl_client.get("/risk-tags").json()
    real_client = httpx.AsyncClient

    def client(**kwargs):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=upstream))
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(gateway.httpx, "AsyncClient", client)
    res = TestClient(gateway.app).get("/risk-tags")
    assert res.status_code == 200
    assert res.json() == upstream

    monkeypatch.setattr(gateway.httpx, "AsyncClient", real_client)
    monkeypatch.setattr(gateway, "RL_API_URL", "http://127.0.0.1:9")
    assert TestClient(gateway.app).get("/risk-tags").status_code == 503
