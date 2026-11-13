"""실제 백테스트 성과 API(GET /performance) 테스트.

임시 CSV(conftest.perf_results)로 4년 연결 계산, 시드 평균, 벤치마크 지표,
Safe-Guard 한도 초과 구간, Gateway 전달을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import gateway
import rl_server
from conftest import BENCHMARK, RETURNS

rl_client = TestClient(rl_server.app)


def _no_model():
    raise FileNotFoundError("모델 파일이 없습니다")


@pytest.fixture
def perf(perf_results, monkeypatch):
    monkeypatch.setattr(rl_server, "get_schedule", _no_model)
    res = rl_client.get("/performance")
    assert res.status_code == 200
    return res.json()


def by_kind(perf: dict) -> dict:
    return {(e["kind"], e["strategy"]): e for e in perf["strategies"]}


def test_performance_identifies_current_model(perf):
    """현재 모델 이름에서 보상 함수·시드를 찾아 해당 결과를 고르는지 검증"""
    assert (perf["strategy"], perf["seed"]) == ("PPO-simple", "42")
    assert perf["data_mode"] == "real"
    assert [w["start"] for w in perf["windows"]] == ["2024-01-02", "2025-01-02"]
    kinds = [e["kind"] for e in perf["strategies"]]
    assert kinds == ["model", "baseline", "benchmark", "variant"]
    assert "투자 조언" in perf["disclaimer"]


def test_performance_chains_windows_into_one_curve(perf):
    """구간별 수익률을 4년 연속 자산가치로 이어 붙이고 낙폭을 계산하는지 검증"""
    returns = np.array(RETURNS[("PPO-simple", "42")])
    value = np.cumprod(1 + returns)
    series = perf["series"]
    assert len(series["dates"]) == 6
    assert series["model"]["cumulative_return"] == pytest.approx(value - 1, abs=1e-6)
    peak = np.maximum.accumulate(np.concatenate([[1.0], value]))[1:]
    assert series["model"]["drawdown"] == pytest.approx(1 - value / peak, abs=1e-6)


def test_performance_uses_seed_mean_for_other_rewards(perf):
    """다른 보상 함수는 시드 평균으로 비교하는지 검증 (특정 시드 선택 편향 방지)"""
    entries = by_kind(perf)
    sharpe = entries[("variant", "PPO-sharpe")]
    assert sharpe["seed"] == "mean"
    # conftest에서 PPO-sharpe 시드 42·43의 지표는 각각 1, 2 → 평균 1.5
    assert sharpe["total"]["sharpe"] == pytest.approx(1.5)
    assert entries[("model", "PPO-simple")]["total"]["sharpe"] == pytest.approx(0)


def test_performance_benchmark_metrics_match_definition(perf):
    """벤치마크(SPY) 지표를 rl-engine과 같은 공식으로 계산하는지 검증"""
    spy = by_kind(perf)[("benchmark", "SPY")]["total"]
    r = pd.Series(BENCHMARK)
    assert spy["cumulative_return"] == pytest.approx(np.prod(1 + r) - 1)
    assert spy["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert (spy["alpha"], spy["beta"], spy["information_ratio"]) == (0.0, 1.0, None)


def test_performance_reports_safeguard_breaches(perf):
    """낙폭이 15%를 넘은 구간을 시작·종료일과 최대 낙폭으로 보고하는지 검증"""
    assert perf["mdd_limit"] == 0.15
    assert len(perf["breaches"]) == 1
    event = perf["breaches"][0]
    assert (event["start"], event["end"], event["days"]) == (
        "2025-01-03",
        "2025-01-06",
        2,
    )
    assert event["max_drawdown"] == pytest.approx(
        max(perf["series"]["model"]["drawdown"])
    )


def test_performance_includes_model_weights_when_available(perf_results, monkeypatch):
    """모델을 쓸 수 있으면 날짜별 목표 비중을 함께 반환하는지 검증"""
    schedule = rl_server.PolicySchedule(
        model_name="fake",
        tickers=["SPY", "TLT"],
        dates=pd.DatetimeIndex(["2025-01-02", "2025-01-03"]),
        weights=np.array([[0.6, 0.4], [0.5, 0.5]]),
    )
    monkeypatch.setattr(rl_server, "get_schedule", lambda: schedule)
    weights = rl_client.get("/performance").json()["weights"]
    assert weights["dates"] == ["2025-01-02", "2025-01-03"]
    assert weights["values"] == [[0.6, 0.4], [0.5, 0.5]]


def test_performance_missing_results_returns_503(perf_results, monkeypatch):
    """현재 모델의 결과가 CSV에 없으면 503과 사유를 반환하는지 검증"""
    monkeypatch.setattr(rl_server, "RL_MODEL_NAME", "ppo_test_w9_none_s1")
    (perf_results / "ppo_test_w9_none_s1.json").write_text(
        '{"variant": "none", "seed": 1}', encoding="utf-8"
    )
    res = rl_client.get("/performance")
    assert res.status_code == 503
    assert "PPO-none" in res.json()["detail"]


def test_gateway_performance_unavailable_returns_503(monkeypatch):
    """RL 서버가 꺼져 있으면 Gateway /performance가 503으로 알리는지 검증"""
    monkeypatch.setattr(gateway, "RL_API_URL", "http://127.0.0.1:9")
    res = TestClient(gateway.app).get("/performance")
    assert res.status_code == 503
