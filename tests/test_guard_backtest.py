"""리스크 태그 Safe-Guard 켬/끔 백테스트 테스트.

CI에는 torch가 없으므로 rl-engine 환경과 같은 규칙(목표 비중 → 회전율 비용 → 다음 날 수익)을
따르는 작은 가짜 환경과 고정 정책으로 폐루프 재생을 검증한다.
"""

import httpx
import numpy as np
import pandas as pd
import pytest
from conftest import risk_tag
from fastapi.testclient import TestClient

import gateway
import rl_server

TICKERS = ["AAA", "BBB", "CCC"]
COST = 0.001  # 회전율당 거래비용


class FakeEnv:
    """rl-engine PortfolioEnv와 같은 순서: 결정 → step에서 다음 거래일 수익 반영."""

    def __init__(self, returns: np.ndarray, dates: pd.DatetimeIndex):
        self.returns, self.dates = returns, dates
        self.n_steps = len(dates)
        self.received = []  # 환경이 실제로 받은 목표 비중

    def reset(self, seed=None):
        self.current_step = 0
        self.weights = np.full(len(TICKERS), 1 / len(TICKERS))
        return self.weights.copy(), {}

    def step(self, action):
        target = np.asarray(action, dtype=float) / np.sum(action)
        self.received.append(target)
        turnover = float(np.abs(target - self.weights).sum())
        self.current_step += 1
        net = float(target @ self.returns[self.current_step]) - turnover * COST
        self.weights = target
        info = {"net_return": net, "turnover": turnover}
        return self.weights.copy(), net, False, False, info


class EqualPolicy:
    def predict(self, obs, deterministic=True):
        return np.ones(len(TICKERS)), None


@pytest.fixture
def policy():
    """AAA가 2025-01-03에 10% 급락하는 5거래일 시장과 동일 비중 정책."""
    dates = pd.bdate_range("2025-01-02", periods=5)
    returns = np.full((5, 3), 0.001)
    returns[1, 0] = -0.10  # 01-02 결정 → 01-03 수익에서 AAA 급락
    envs = []

    def make_env():
        envs.append(FakeEnv(returns, dates))
        return envs[-1]

    fake = rl_server.Policy(
        model_name="fake",
        tickers=TICKERS,
        max_weight=0.4,
        model=EqualPolicy(),
        make_env=make_env,
        action_to_weights=lambda action, cap: np.asarray(action) / np.sum(action),
        weights_to_action=lambda weights: np.asarray(weights),
    )
    return fake, envs


def test_replay_without_guard_matches_model_weights(policy):
    """Safe-Guard 없이 재생하면 결정일마다 모델 비중 그대로, 일별 성과는 거래일 수 − 1개인지 검증"""
    fake, _ = policy
    dates, weights, daily = rl_server.replay(fake)
    assert len(dates) == 5 and len(daily) == 4
    np.testing.assert_allclose(weights, 1 / 3)
    assert daily.index[0] == pd.Timestamp("2025-01-03")
    assert daily["net_return"].iloc[0] == pytest.approx(0.001 * 2 / 3 - 0.10 / 3)


def test_guard_backtest_feeds_adjusted_weights_back(policy, write_tags):
    """하락 직전 공개된 태그로 줄인 비중이 실제로 체결되고(폐루프), 켠 쪽 성과가 나아지는지 검증"""
    fake, envs = policy
    # 01-02 장 마감(뉴욕 16시) 공개, 점수 0.8 → AAA 비중 × (1 − 0.5 × 0.8)
    write_tags(risk_tag("AAA", 0.8, "2025-01-02T16:00:00-05:00", days=1))
    result = rl_server.build_guard_backtest(fake, rl_server.get_risk_tags())

    guarded_env = envs[-1]
    assert guarded_env.received[0][0] == pytest.approx(1 / 3 * 0.6)
    assert result["adjusted_days"] == 1
    assert result["average_moved"] == pytest.approx(1 / 3 * 0.4)
    model, guarded = result["metrics"]["model"], result["metrics"]["guarded"]
    assert guarded["cumulative_return"] > model["cumulative_return"]
    assert result["difference"]["mdd"] < 0  # 낙폭 감소
    assert "alpha" not in model and model["avg_turnover"] is not None
    assert len(result["series"]["dates"]) == len(
        result["series"]["guarded"]["drawdown"]
    )


def test_guard_backtest_without_live_tags_is_identical(policy, write_tags):
    """평가 구간에 유효한 태그가 없으면 두 재생 결과가 완전히 같은지 검증"""
    fake, _ = policy
    write_tags(risk_tag("AAA", 0.9, "2024-06-01T16:00:00-04:00", days=3))
    result = rl_server.build_guard_backtest(fake, rl_server.get_risk_tags())
    assert result["adjusted_days"] == 0
    assert all(v in (0.0, None) for v in result["difference"].values())


def test_guard_backtest_reports_match_with_published_results(
    policy, write_tags, monkeypatch
):
    """끈 쪽 재생을 rl-engine 공개 일별 수익률과 대조한 결과(일치 일수·최대 오차)를 보고하는지 검증"""
    fake, _ = policy
    _, _, daily = rl_server.replay(fake)
    monkeypatch.setattr(rl_server, "_published_daily", lambda: daily["net_return"])
    write_tags(risk_tag("AAA", 0.8, "2025-01-02T16:00:00-05:00"))
    result = rl_server.build_guard_backtest(fake, rl_server.get_risk_tags())
    assert result["reproduces_published"] == {"matched_days": 4, "max_abs_diff": 0.0}


def test_guard_backtest_endpoint(policy, write_tags, monkeypatch):
    """/risk-tags/backtest가 모델이 없으면 503, 있으면 비교 결과를 반환하는지 검증"""
    client = TestClient(rl_server.app)

    def broken():
        raise FileNotFoundError("모델 없음")

    monkeypatch.setattr(rl_server, "get_policy", broken)
    assert client.get("/risk-tags/backtest").status_code == 503

    fake, _ = policy
    monkeypatch.setattr(rl_server, "get_policy", lambda: fake)
    write_tags(risk_tag("AAA", 0.8, "2025-01-02T16:00:00-05:00"))
    data = client.get("/risk-tags/backtest").json()
    assert data["model_name"] == "fake"
    assert set(data["metrics"]) == {"model", "guarded"}
    assert "투자 조언" in data["disclaimer"]


def test_gateway_forwards_guard_backtest(monkeypatch):
    """Gateway /risk-tags/backtest가 RL 서버 응답을 그대로 전달하고, 꺼져 있으면 503인지 검증"""
    real = httpx.AsyncClient
    payload = {
        "status": "ok",
        "model_name": "fake",
        "period": {"start": "2025-01-03", "end": "2025-01-08"},
        "method": "폐루프 재생",
        "adjusted_days": 1,
        "metrics": {"model": {"mdd": 0.03}, "guarded": {"mdd": 0.02}},
        "difference": {"mdd": -0.01},
        "series": {"dates": []},
        "reproduces_published": None,
        "limitation": "1년 구간",
        "disclaimer": "교육 목적",
        "extra": "그대로 전달",
    }

    def make(handler):
        def client(**kwargs):
            return real(transport=httpx.MockTransport(handler), **kwargs)

        return client

    monkeypatch.setattr(
        gateway.httpx,
        "AsyncClient",
        make(lambda r: httpx.Response(200, json=payload)),
    )
    client = TestClient(gateway.app)
    res = client.get("/risk-tags/backtest")
    assert res.status_code == 200 and res.json() == payload

    def refuse(request):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(gateway.httpx, "AsyncClient", make(refuse))
    assert client.get("/risk-tags/backtest").status_code == 503
