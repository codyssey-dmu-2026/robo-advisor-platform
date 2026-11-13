"""여러 테스트 파일이 함께 쓰는 준비물: rl-engine 백테스트 결과 CSV를 흉내 낸 임시 폴더."""

import json

import pandas as pd
import pytest

import rl_server

# 구간 2개 × 3거래일. 현재 모델(PPO-simple 시드 42)은 둘째 구간에서 낙폭이 15%를 넘는다
DATES = {
    1: ["2024-01-02", "2024-01-03", "2024-01-04"],
    2: ["2025-01-02", "2025-01-03", "2025-01-06"],
}
RETURNS = {
    ("PPO-simple", "42"): [0.10, -0.05, 0.02, 0.01, -0.20, 0.05],
    ("PPO-sharpe", "42"): [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    ("PPO-sharpe", "43"): [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    ("EqualWeight", "-"): [0.02, 0.00, 0.01, 0.00, -0.01, 0.02],
}
BENCHMARK = [0.01, -0.02, 0.03, 0.00, -0.05, 0.04]
METRICS = list(rl_server.METRIC_COLUMNS)


def _metrics(value: float) -> dict:
    """지표 값은 계산 검증 대상이 아니므로 전략마다 다른 상수로 채운다."""
    return {col: value for col in METRICS}


@pytest.fixture
def perf_results(tmp_path, monkeypatch):
    (tmp_path / "ppo_test_w2_simple_s42.json").write_text(
        json.dumps({"variant": "simple", "seed": 42}), encoding="utf-8"
    )
    daily, windows, walkforward = [], [], []
    for n, ((strategy, seed), returns) in enumerate(RETURNS.items()):
        i = 0
        for window, dates in DATES.items():
            for date in dates:
                daily.append(
                    {
                        "date": date,
                        "window": window,
                        "strategy": strategy,
                        "seed": seed,
                        "net_return": returns[i],
                        "benchmark_return": BENCHMARK[i],
                    }
                )
                i += 1
            windows.append(
                {"window": window, "strategy": strategy, "seed": seed}
                | _metrics(n + window / 10)
            )
        walkforward.append({"strategy": strategy, "seed": seed} | _metrics(n))
    pd.DataFrame(daily).to_csv(tmp_path / "eval_test_daily.csv", index=False)
    pd.DataFrame(windows).to_csv(tmp_path / "eval_test_windows.csv", index=False)
    pd.DataFrame(walkforward).to_csv(
        tmp_path / "eval_test_walkforward.csv", index=False
    )

    monkeypatch.setattr(rl_server, "RL_MODELS_DIR", tmp_path)
    monkeypatch.setattr(rl_server, "RL_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rl_server, "RL_MODEL_NAME", "ppo_test_w2_simple_s42")
    rl_server.get_performance.cache_clear()
    yield tmp_path
    rl_server.get_performance.cache_clear()
