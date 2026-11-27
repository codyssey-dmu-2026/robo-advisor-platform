"""rl-engine 학습 결과를 MLflow에 기록하고 모델 60개를 모델 저장소에 등록한다.

docker compose의 mlflow-bootstrap이 처음 한 번 실행한다(이미 등록돼 있으면 건너뜀).
서빙 모델(SERVING_MODEL)에는 항상 `production` 별칭을 다시 붙여, .env의 RL_MODEL_NAME을 바꾸면
다음 실행에서 서빙 버전도 바뀐다.

- 실험 `rl-engine-test`  : 최종 모델 60개 (설정, 테스트 구간 12개 지표, 학습 곡선, 체크포인트)
- 실험 `rl-engine-val`   : 보상 함수 선택용 학습 60개 (설정, 검증 구간 지표, 학습 곡선)
- 모델 `ppo-portfolio`   : 최종 모델 60개를 버전으로 등록

    MLFLOW_TRACKING_URI=http://mlflow:5000 python -m mlops.register_experiments
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

from mlops.registry import MODEL_NAME, PRODUCTION

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("RL_MODELS_DIR", ROOT.parent / "rl-engine" / "models"))
RESULTS_DIR = Path(os.getenv("RL_RESULTS_DIR", MODELS_DIR))
CURVES = Path(os.getenv("RL_LEARNING_CURVES", ROOT / "data" / "learning_curves.csv"))
SERVING_MODEL = os.getenv("SERVING_MODEL", "ppo_test_w4_simple_s42")
SOURCE = os.getenv("REGISTRY_SOURCE", "rl-engine 05579d2 · platform Release models-v1")
FORCE = os.getenv("REGISTRY_FORCE", "0") == "1"
# 마지막으로 production에 반영한 설정값(RL_MODEL_NAME)을 등록 모델 태그로 남긴다
APPLIED = "serving_model_config"

RUN = re.compile(
    r"^ppo_(?P<stage>test|val)_w(?P<window>\d+)_(?P<variant>.+)_s(?P<seed>\d+)$"
)
# 모델 메타 중 MLflow 파라미터로 남길 항목 (개인 경로가 든 data_dir 등은 제외)
PARAM_KEYS = [
    "stage",
    "window",
    "variant",
    "reward_type",
    "mdd_lambda",
    "seed",
    "timesteps",
    "gamma",
    "lookback",
    "max_weight",
    "mdd_limit",
    "obs_dim",
    "train_start",
    "val_start",
    "train_end",
    "test_end",
]
TEST_METRICS = [
    "cum_return",
    "cagr",
    "ann_vol",
    "var_95",
    "cvar_95",
    "max_mdd",
    "sharpe",
    "sortino",
    "calmar",
    "alpha",
    "beta",
    "information_ratio",
    "win_rate",
    "avg_turnover",
]
VAL_METRICS = ["cum_return", "ann_vol", "sharpe", "max_mdd", "avg_turnover"]
CURVE_METRICS = ["ep_rew_mean", "ep_len_mean"]


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _results(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_DIR / name, dtype={"seed": str})
    return frame.assign(seed=frame["seed"].fillna("-"))


def _curve(curves: pd.DataFrame, name: str) -> dict:
    curve = curves[curves.run == name] if not curves.empty else curves
    return {
        key: [
            (int(step), v)
            for step, value in zip(curve["step"], curve[key])
            if (v := _number(value)) is not None
        ]
        for key in CURVE_METRICS
        if key in curve
    }


def _metrics(rows: pd.DataFrame, keys: list[str]) -> dict:
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {k: v for k in keys if (v := _number(row.get(k))) is not None}


def build_payloads() -> list[dict]:
    """MLflow에 보낼 실행 기록(설정·지표·학습 곡선·체크포인트 경로)을 만든다.

    최종 모델은 모델 메타(.json)와 체크포인트(.zip)에서, 선택용(val) 학습은 체크포인트가
    배포되지 않으므로 rl-engine 결과 CSV(compare_val.csv)와 학습 곡선에서 만든다.
    """
    test = _results("eval_test_windows.csv")
    val = _results("compare_val.csv")
    curves = pd.read_csv(CURVES) if CURVES.is_file() else pd.DataFrame()
    payloads = []
    for meta_path in sorted(MODELS_DIR.glob("ppo_test_w*_s*.json")):
        name = meta_path.stem
        match = RUN.match(name)
        if not match:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        window, seed = int(match["window"]), match["seed"]
        rows = test[
            (test.strategy == f"PPO-{match['variant']}")
            & (test.seed == seed)
            & (test.window == window)
        ]
        checkpoint = meta_path.with_suffix(".zip")
        payloads.append(
            {
                "name": name,
                "stage": "test",
                "params": {k: str(meta[k]) for k in PARAM_KEYS if k in meta},
                "tags": {
                    "model_name": name,
                    "variant": match["variant"],
                    "window": str(window),
                    "seed": seed,
                    "source": SOURCE,
                },
                "metrics": _metrics(rows, TEST_METRICS),
                "curves": _curve(curves, name),
                "checkpoint": checkpoint if checkpoint.is_file() else None,
                "meta": meta_path,
            }
        )
    ppo = val[val.strategy.str.startswith("PPO-")]
    for _, row in ppo.sort_values(["window", "strategy", "seed"]).iterrows():
        variant = row["strategy"].removeprefix("PPO-")
        name = f"ppo_val_w{int(row['window'])}_{variant}_s{row['seed']}"
        params = {
            "stage": "val",
            "window": str(int(row["window"])),
            "variant": variant,
            "reward_type": str(row.get("reward_type")),
            "seed": str(row["seed"]),
            "timesteps": str(int(row["timesteps"])),
        }
        if variant.startswith("mdd"):
            params["mdd_lambda"] = str(row.get("mdd_lambda"))
        payloads.append(
            {
                "name": name,
                "stage": "val",
                "params": params,
                "tags": {
                    "model_name": name,
                    "variant": variant,
                    "window": params["window"],
                    "seed": params["seed"],
                    "source": SOURCE,
                },
                "metrics": _metrics(pd.DataFrame([row]), VAL_METRICS),
                "curves": _curve(curves, name),
                "checkpoint": None,
                "meta": None,
            }
        )
    return payloads


def _log(client, experiment_id: str, payload: dict) -> str:
    from mlflow.entities import Metric, Param, RunTag

    run = client.create_run(
        experiment_id, run_name=payload["name"], tags=payload["tags"]
    )
    now = int(time.time() * 1000)
    params = [Param(k, v) for k, v in payload["params"].items()]
    metrics = [Metric(k, v, now, 0) for k, v in payload["metrics"].items()]
    for key, points in payload["curves"].items():
        metrics += [Metric(f"train/{key}", v, now, step) for step, v in points]
    # log_batch는 한 번에 지표 1,000개까지 받는다
    for start in range(0, max(len(metrics), 1), 1000):
        client.log_batch(
            run.info.run_id,
            metrics=metrics[start : start + 1000],
            params=params if start == 0 else [],
            tags=(
                [RunTag("mlflow.note.content", "rl-engine 학습 결과 (platform 등록)")]
                if start == 0
                else []
            ),
        )
    if payload["checkpoint"] is not None:
        client.log_artifact(run.info.run_id, str(payload["checkpoint"]), "checkpoint")
        client.log_artifact(run.info.run_id, str(payload["meta"]), "checkpoint")
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def _logged_names(client, experiment_id: str) -> set:
    """실험에 이미 기록된 실행 이름 (다시 실행해도 중복으로 쌓이지 않게)."""
    runs = client.search_runs([experiment_id], max_results=5000)
    return {r.data.tags.get("model_name") for r in runs}


def register(client, payloads: list[dict], registered: dict) -> dict:
    """빠진 실행만 기록하고 빠진 최종 모델만 등록한다. 모델 이름 → 버전 번호."""
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(
            MODEL_NAME,
            description="rl-engine PPO 포트폴리오 모델 (Walk-Forward 최종 학습 60개)",
        )
    except MlflowException:
        pass  # 이미 있음
    versions = dict(registered)
    for stage in ("test", "val"):
        experiment = f"rl-engine-{stage}"
        found = client.get_experiment_by_name(experiment)
        experiment_id = (
            found.experiment_id if found else client.create_experiment(experiment)
        )
        logged = _logged_names(client, experiment_id)
        for payload in (p for p in payloads if p["stage"] == stage):
            name = payload["name"]
            if name in versions or (payload["checkpoint"] is None and name in logged):
                continue
            run_id = _log(client, experiment_id, payload)
            if payload["checkpoint"] is None:
                continue
            run = client.get_run(run_id)
            sharpe = payload["metrics"].get("sharpe")
            version = client.create_model_version(
                MODEL_NAME,
                source=f"{run.info.artifact_uri}/checkpoint",
                run_id=run_id,
                tags=payload["tags"],
                description=name
                + (f" (테스트 샤프 {sharpe:.3f})" if sharpe is not None else ""),
            )
            versions[name] = version.version
    return versions


def existing_versions(client) -> dict:
    """이미 등록된 버전: 모델 이름 → 버전 번호."""
    found = client.search_model_versions(f"name='{MODEL_NAME}'")
    return {
        v.tags.get("model_name"): v.version for v in found if v.tags.get("model_name")
    }


def main() -> None:
    from mlflow import MlflowClient

    client = MlflowClient()
    payloads = build_payloads()
    if not payloads:
        sys.exit(f"[mlflow-bootstrap] 모델 메타 파일이 없습니다: {MODELS_DIR}")
    versions = {} if FORCE else existing_versions(client)
    before = len(versions)
    versions = register(client, payloads, versions)
    print(
        f"[mlflow-bootstrap] 실행 기록 {len(payloads)}개 확인, 모델 버전 {len(versions)}개 "
        f"(이번에 새로 등록 {len(versions) - before}개)"
    )
    if SERVING_MODEL not in versions:
        sys.exit(
            f"[mlflow-bootstrap] 서빙 모델이 등록돼 있지 않습니다: {SERVING_MODEL}"
        )
    model = client.get_registered_model(MODEL_NAME)
    applied = model.tags.get(APPLIED)
    if should_move_production(applied, SERVING_MODEL, PRODUCTION in model.aliases):
        version = versions[SERVING_MODEL]
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION, version)
        client.set_registered_model_tag(MODEL_NAME, APPLIED, SERVING_MODEL)
        print(f"[mlflow-bootstrap] {MODEL_NAME}@{PRODUCTION} → v{version}")
    else:
        # 설정이 그대로면 승격(mlops.promote)으로 옮긴 운영 버전을 되돌리지 않는다
        current = model.aliases[PRODUCTION]
        print(
            f"[mlflow-bootstrap] 설정 변경 없음: "
            f"{MODEL_NAME}@{PRODUCTION} v{current} 유지"
        )


def should_move_production(applied: str | None, serving: str, has_alias: bool) -> bool:
    """production 별칭을 설정값(RL_MODEL_NAME) 모델로 옮길지.

    처음이거나 설정값이 바뀌었을 때만 옮긴다. 설정이 그대로면 사람이 승격한 버전을 유지한다.
    """
    return not has_alias or applied != serving


if __name__ == "__main__":
    main()
