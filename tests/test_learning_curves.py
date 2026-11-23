"""학습 곡선 API 테스트: 실제 변환본 검사, 요약 계산, 파일 없음 503, Gateway 전달."""

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import gateway
import rl_server


@pytest.fixture(autouse=True)
def fresh_cache():
    rl_server.get_learning_curves.cache_clear()
    yield
    rl_server.get_learning_curves.cache_clear()


def tiny_curves() -> pd.DataFrame:
    rows = []
    for seed in (42, 43):
        for i, step in enumerate(range(2048, 2048 * 11, 2048)):
            rows.append(
                {
                    "run": f"ppo_test_w4_simple_s{seed}",
                    "stage": "test",
                    "window": 4,
                    "variant": "simple",
                    "seed": seed,
                    "attempt": "PPO_1",
                    "step": step,
                    "ep_rew_mean": i * 0.1,
                    "ep_len_mean": 200 + i,
                }
            )
    return pd.DataFrame(rows)


def test_shipped_learning_curves_meet_minimum_steps():
    """레포에 넣은 학습 곡선이 보상 5종 × 구간 4 × seed 3 × 2단계이고 모두 10만 스텝 이상인지 검증"""
    data = rl_server.get_learning_curves()
    runs = data["runs"]
    assert data["run_count"] == len(runs) == 120
    assert {r["variant"] for r in runs} == {
        "simple",
        "sharpe",
        "mdd_l0.5",
        "mdd_l1.0",
        "mdd_l5.0",
    }
    assert {(r["stage"], r["window"]) for r in runs} == {
        (s, w) for s in ("test", "val") for w in (1, 2, 3, 4)
    }
    assert min(r["total_steps"] for r in runs) >= 100_000
    # 현재 서빙 모델의 곡선이 들어 있다
    assert any(r["run"] == data["current_model"] for r in runs)


def test_learning_curve_summary_uses_first_and_last_five_points():
    """시작·마지막 보상이 처음·마지막 5개 기록 평균이고 실행별로 스텝 순서가 유지되는지 검증"""
    data = rl_server.build_learning_curves(tiny_curves().sample(frac=1, random_state=0))
    run = next(r for r in data["runs"] if r["seed"] == 42)
    assert run["steps"] == sorted(run["steps"])
    assert run["start_reward"] == pytest.approx(0.2)  # (0+0.1+0.2+0.3+0.4)/5
    assert run["end_reward"] == pytest.approx(0.7)  # (0.5+…+0.9)/5
    assert run["end_episode_length"] == pytest.approx(207)
    assert "비교할 수 없습니다" in data["note"]


def test_learning_curves_missing_file_is_503(tmp_path, monkeypatch):
    """학습 곡선 파일이 없으면 /learning-curves가 503과 사유를 반환하는지 검증"""
    monkeypatch.setattr(rl_server, "RL_LEARNING_CURVES", tmp_path / "none.csv")
    res = TestClient(rl_server.app).get("/learning-curves")
    assert res.status_code == 503
    assert "학습 곡선" in res.json()["detail"]


def test_gateway_forwards_learning_curves(monkeypatch):
    """Gateway /learning-curves가 RL 서버 응답을 그대로 전달하는지 검증"""
    payload = rl_server.build_learning_curves(tiny_curves())
    real = httpx.AsyncClient

    def client(**kwargs):
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        return real(transport=transport, **kwargs)

    monkeypatch.setattr(gateway.httpx, "AsyncClient", client)
    res = TestClient(gateway.app).get("/learning-curves")
    assert res.status_code == 200
    assert res.json()["run_count"] == 2
