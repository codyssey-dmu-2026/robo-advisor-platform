"""MLOps(MLflow) 테스트: 모델 URI, 등록할 실행 기록 만들기, 서빙 모델 출처와 대체 동작.

CI에는 mlflow가 없으므로 MLflow 서버와 주고받는 부분은 docker compose에서 확인하고,
여기서는 보낼 내용을 만드는 로직과 서빙 서버의 대체 동작을 검증한다.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import rl_server
from mlops import register_experiments as reg
from mlops.registry import parse_uri


def test_parse_model_uri():
    """models:/<이름>@<별칭> 형식만 받고 나머지는 거절하는지 검증"""
    assert parse_uri("models:/ppo-portfolio@production") == (
        "ppo-portfolio",
        "production",
    )
    for bad in ("models:/ppo-portfolio", "runs:/abc/model", "models:/a@b c"):
        with pytest.raises(ValueError):
            parse_uri(bad)


@pytest.fixture
def rl_results(tmp_path, monkeypatch):
    """모델 메타 1개(체크포인트 포함), 테스트·검증 결과 CSV, 학습 곡선을 임시 폴더에 만든다."""
    meta = {
        "stage": "test",
        "window": 4,
        "variant": "simple",
        "reward_type": "simple",
        "mdd_lambda": 1.0,
        "seed": 42,
        "timesteps": 100000,
        "lookback": 20,
        "train_start": "2018-01-01",
        "data_dir": "C:\\Users\\someone\\private\\data",
    }
    (tmp_path / "ppo_test_w4_simple_s42.json").write_text(json.dumps(meta))
    (tmp_path / "ppo_test_w4_simple_s42.zip").write_bytes(b"checkpoint")
    pd.DataFrame(
        [
            {
                "window": 4,
                "strategy": "PPO-simple",
                "seed": "42",
                "sharpe": 2.15,
                "cum_return": 0.445,
                "max_mdd": 0.168,
                "alpha": float("nan"),
            }
        ]
    ).to_csv(tmp_path / "eval_test_windows.csv", index=False)
    pd.DataFrame(
        [
            {"window": 4, "strategy": "EqualWeight", "seed": None, "sharpe": 1.0},
            {
                "window": 4,
                "strategy": "PPO-mdd_l1.0",
                "seed": "43",
                "reward_type": "mdd",
                "mdd_lambda": 1.0,
                "timesteps": 100000,
                "sharpe": 0.4,
                "cum_return": 0.02,
            },
        ]
    ).to_csv(tmp_path / "compare_val.csv", index=False)
    curves = tmp_path / "curves.csv"
    pd.DataFrame(
        [
            {"run": run, "step": step, "ep_rew_mean": step / 1e5, "ep_len_mean": 200}
            for run in ("ppo_test_w4_simple_s42", "ppo_val_w4_mdd_l1.0_s43")
            for step in (2048, 4096, 100352)
        ]
    ).to_csv(curves, index=False)
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(reg, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(reg, "CURVES", curves)
    return tmp_path


def test_payloads_record_params_metrics_curves_without_private_paths(rl_results):
    """최종 모델 기록에 설정·테스트 지표·학습 곡선·체크포인트가 들어가고 개인 경로는 빠지는지 검증"""
    payloads = {p["name"]: p for p in reg.build_payloads()}
    test = payloads["ppo_test_w4_simple_s42"]
    assert test["stage"] == "test"
    assert "data_dir" not in test["params"]
    assert test["params"]["train_start"] == "2018-01-01"
    assert test["metrics"] == {"sharpe": 2.15, "cum_return": 0.445, "max_mdd": 0.168}
    assert [step for step, _ in test["curves"]["ep_rew_mean"]] == [2048, 4096, 100352]
    assert test["checkpoint"].name == "ppo_test_w4_simple_s42.zip"
    assert test["tags"]["model_name"] == "ppo_test_w4_simple_s42"


def test_validation_runs_come_from_results_without_checkpoint(rl_results):
    """선택용(val) 학습은 체크포인트 없이 결과 CSV·학습 곡선만으로 기록되고 기준선은 빠지는지 검증"""
    payloads = {p["name"]: p for p in reg.build_payloads()}
    assert set(payloads) == {"ppo_test_w4_simple_s42", "ppo_val_w4_mdd_l1.0_s43"}
    val = payloads["ppo_val_w4_mdd_l1.0_s43"]
    assert val["checkpoint"] is None and val["stage"] == "val"
    assert val["params"]["mdd_lambda"] == "1.0"
    assert val["metrics"]["sharpe"] == 0.4
    assert len(val["curves"]["ep_rew_mean"]) == 3


def test_model_source_defaults_to_file(monkeypatch):
    """RL_MODEL_URI가 없으면 파일 모델(RL_MODELS_DIR/RL_MODEL_NAME)을 쓰는지 검증"""
    monkeypatch.setattr(rl_server, "RL_MODEL_URI", "")
    source = rl_server.model_source()
    assert source["source"] == "file"
    assert source["name"] == rl_server.RL_MODEL_NAME


def test_model_source_falls_back_when_registry_unavailable(monkeypatch):
    """MLflow 모델 저장소를 쓸 수 없으면 파일 모델로 대체하고 사유를 /health에 남기는지 검증"""
    import mlops.registry

    def unavailable(uri, destination):
        raise ConnectionError("mlflow:5000에 연결할 수 없음")

    monkeypatch.setattr(rl_server, "RL_MODEL_URI", "models:/ppo-portfolio@production")
    monkeypatch.setattr(rl_server, "_REGISTRY", {})
    monkeypatch.setattr(mlops.registry, "resolve_model", unavailable)
    source = rl_server.model_source()
    assert source["source"] == "file"
    assert "ConnectionError" in source["fallback_reason"]

    monkeypatch.setattr(
        rl_server, "get_schedule", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    health = TestClient(rl_server.app).get("/health").json()
    assert health["model_source"]["uri"] == "models:/ppo-portfolio@production"
    assert "dir" not in health["model_source"]


def test_model_source_uses_registry_version(monkeypatch, tmp_path):
    """모델 저장소에서 찾은 버전의 이름·폴더를 쓰고, 한 번 찾으면 다시 묻지 않는지 검증"""
    import mlops.registry

    calls = []

    def resolved(uri, destination):
        calls.append(uri)
        return {
            "name": "ppo_test_w4_simple_s42",
            "dir": tmp_path,
            "source": "mlflow",
            "uri": uri,
            "version": "58",
            "alias": "production",
        }

    monkeypatch.setattr(rl_server, "RL_MODEL_URI", "models:/ppo-portfolio@production")
    monkeypatch.setattr(rl_server, "_REGISTRY", {})
    monkeypatch.setattr(mlops.registry, "resolve_model", resolved)
    assert rl_server.model_source()["version"] == "58"
    assert rl_server.model_source()["dir"] == tmp_path
    assert len(calls) == 1
    assert rl_server.model_source_info()["source"] == "mlflow"
