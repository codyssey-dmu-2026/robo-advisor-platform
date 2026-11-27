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


# ---------------- 2부: 재학습 파이프라인 ----------------
from datetime import datetime, timezone  # noqa: E402

from mlops import retrain  # noqa: E402


def test_parse_variant():
    """보상 함수 이름에서 rl-engine 학습 설정(reward_type, λ)을 읽는지 검증"""
    assert retrain.parse_variant("simple") == ("simple", 1.0)
    assert retrain.parse_variant("mdd_l0.5") == ("mdd", 0.5)
    with pytest.raises(ValueError):
        retrain.parse_variant("unknown")


@pytest.mark.parametrize(
    "candidate, reference, ok, phrase",
    [
        (0.40, 0.367, True, "≥ 원래"),
        (0.10, 0.367, False, "< 원래"),
        (None, 0.367, False, "계산하지 못함"),
        (0.40, None, False, "사람이 검토"),
    ],
)
def test_promotion_uses_validation_sharpe_only(candidate, reference, ok, phrase):
    """승격 권장은 검증 샤프로만 정하고, 비교 기준이 없으면 사람 검토로 넘기는지 검증"""
    recommended, reason = retrain.recommend(candidate, reference, 0.0)
    assert recommended is ok
    assert phrase in reason


def test_next_run_is_today_or_tomorrow():
    """다음 실행 시각이 오늘 그 시각(아직 안 지났으면) 또는 내일인지 검증"""
    morning = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)
    assert retrain.next_run(morning, "19:00") == morning.replace(hour=19)
    evening = datetime(2026, 11, 27, 20, 0, tzinfo=timezone.utc)
    assert retrain.next_run(evening, "19:00").day == 28


def test_fingerprint_changes_with_data_and_window_follows_manifest(tmp_path):
    """데이터 파일이 바뀌면 지문이 바뀌고, 마지막 Walk-Forward 구간을 고르는지 검증"""
    manifest = {"windows": [{"window": 1}, {"window": 4}, {"window": 2}]}
    (tmp_path / "walk_forward_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "returns.parquet").write_bytes(b"returns-v1")
    (tmp_path / "features_window4.parquet").write_bytes(b"features")
    assert retrain.latest_window(tmp_path) == 4
    before = retrain.data_fingerprint(tmp_path, 4)
    assert before == retrain.data_fingerprint(tmp_path, 4)
    (tmp_path / "returns.parquet").write_bytes(b"returns-v2")
    assert retrain.data_fingerprint(tmp_path, 4) != before


@pytest.mark.parametrize(
    "applied, serving, has_alias, move",
    [
        (None, "ppo_test_w4_simple_s42", False, True),  # 처음
        ("ppo_test_w4_simple_s42", "ppo_test_w4_simple_s42", True, False),  # 승격 유지
        ("ppo_test_w4_simple_s42", "ppo_test_w4_sharpe_s42", True, True),  # 설정 변경
        ("ppo_test_w4_simple_s42", "ppo_test_w4_simple_s42", False, True),  # 별칭 없음
    ],
)
def test_bootstrap_keeps_promoted_version_unless_config_changes(
    applied, serving, has_alias, move
):
    """설정값(RL_MODEL_NAME)이 그대로면 사람이 승격한 운영 버전을 되돌리지 않는지 검증"""
    assert reg.should_move_production(applied, serving, has_alias) is move


def test_model_registry_endpoint(monkeypatch):
    """/model-registry가 MLflow 미사용·연결 실패·정상(재시작 필요 표시) 경우를 구분하는지 검증"""
    import mlops.registry

    client = TestClient(rl_server.app)
    monkeypatch.setattr(rl_server, "RL_MODEL_URI", "")
    assert client.get("/model-registry").json()["available"] is False

    monkeypatch.setattr(rl_server, "RL_MODEL_URI", "models:/ppo-portfolio@production")
    monkeypatch.setattr(
        rl_server, "_REGISTRY", {"name": "m", "source": "mlflow", "version": "58"}
    )

    def down(name):
        raise ConnectionError("down")

    monkeypatch.setattr(mlops.registry, "summarize_aliases", down)
    data = client.get("/model-registry").json()
    assert data["available"] is False and "ConnectionError" in data["reason"]

    def summary(name):
        return {
            "registered_model": name,
            "version_count": 61,
            "aliases": {
                "production": {"version": "61"},
                "candidate": {"version": "61"},
            },
        }

    monkeypatch.setattr(mlops.registry, "summarize_aliases", summary)
    data = client.get("/model-registry").json()
    assert data["available"] is True and data["version_count"] == 61
    assert data["restart_needed"] is True  # 운영 v61, 서빙 중 v58
