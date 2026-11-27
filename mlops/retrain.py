"""재학습 파이프라인: 최신 데이터로 서빙 설정을 다시 학습하고 MLflow에 후보로 등록한다.

흐름 (데이터가 바뀌었을 때만 실행)
  1. 데이터 지문(sha256) 확인: 지난 재학습과 같으면 건너뜀
  2. 선택용 학습: 학습 구간 중 검증 구간(마지막 6개월)을 빼고 학습 → 검증 구간 평가
  3. 승격 판단: 같은 설정의 원래 선택용 학습(rl-engine 결과)보다 검증 샤프가 나빠지지 않았는지
  4. 최종 학습: 검증 구간까지 포함해 다시 학습한 서빙용(test 단계) 모델
  5. MLflow 실험 `retraining`에 기록하고 모델 저장소에 `candidate` 별칭으로 등록
승격(production 이동)은 자동으로 하지 않는다. 사람이 확인한 뒤 `python -m mlops.promote`.
테스트 구간 성과는 기록만 하고 판단에 쓰지 않는다(테스트 결과로 고르면 미래 정보 누수).

rl-engine 코드는 수정하지 않고 학습 함수(train.train_one)를 그대로 호출한다.
모델·학습 기록 저장 위치만 실행 중에 임시 폴더로 바꿔 rl-engine 폴더에 파일을 남기지 않는다.

    python -m mlops.retrain --once        # 한 번 실행
    python -m mlops.retrain --schedule    # 매일 RETRAIN_AT_UTC에 실행 (기본 19:00 UTC)
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mlops.registry import CANDIDATE, MODEL_NAME, PRODUCTION

RL_REPO_PATH = Path(os.getenv("RL_REPO_PATH", "/opt/rl-engine"))
RL_DATA_DIR = Path(os.getenv("RL_DATA_DIR", "/pipeline-data/processed"))
TIMESTEPS = int(os.getenv("RETRAIN_TIMESTEPS", "100000"))  # 과제 최소 학습 스텝
MIN_GAIN = float(
    os.getenv("RETRAIN_MIN_GAIN", "0.0")
)  # 검증 샤프가 이만큼 이상 나아야 권장
RUN_AT = os.getenv("RETRAIN_AT_UTC", "19:00")  # 매일 실행 시각 (UTC)
EXPERIMENT = "retraining"


LOOKBACK = int(os.getenv("RETRAIN_LOOKBACK", "20"))  # rl-engine 기본 관측 윈도우


def parse_variant(variant: str) -> tuple[str, float]:
    """보상 함수 이름 → (reward_type, mdd_lambda). 예: 'mdd_l0.5' → ('mdd', 0.5)."""
    if variant.startswith("mdd_l"):
        return "mdd", float(variant[len("mdd_l") :])
    if variant in ("simple", "sharpe"):
        return variant, 1.0
    raise ValueError(f"알 수 없는 보상 함수: {variant}")


def latest_window(data_dir: Path) -> int:
    """data-pipeline이 만든 Walk-Forward 구간 중 마지막 번호 (구간이 추가되면 자동으로 따라감)."""
    manifest = json.loads((data_dir / "walk_forward_manifest.json").read_text("utf-8"))
    return max(int(w["window"]) for w in manifest["windows"])


def data_fingerprint(data_dir: Path, window: int) -> str:
    """학습에 쓰는 파일들의 sha256. 같으면 데이터가 바뀌지 않은 것."""
    digest = hashlib.sha256()
    for name in (
        "walk_forward_manifest.json",
        "returns.parquet",
        f"features_window{window}.parquet",
    ):
        digest.update(name.encode())
        digest.update((data_dir / name).read_bytes())
    return digest.hexdigest()


def recommend(
    candidate_sharpe: float | None, reference_sharpe: float | None, min_gain: float
) -> tuple[bool, str]:
    """검증 구간 샤프로 승격 권장 여부를 정한다 (테스트 구간은 쓰지 않음)."""
    if candidate_sharpe is None:
        return False, "후보의 검증 샤프를 계산하지 못함"
    if reference_sharpe is None:
        return False, "비교할 원래 선택용 학습 기록이 없음 → 사람이 검토"
    gain = candidate_sharpe - reference_sharpe
    if gain >= min_gain:
        return (
            True,
            f"검증 샤프 {candidate_sharpe:.3f} ≥ 원래 {reference_sharpe:.3f} (+{gain:.3f})",
        )
    return (
        False,
        f"검증 샤프 {candidate_sharpe:.3f} < 원래 {reference_sharpe:.3f} ({gain:+.3f})",
    )


def next_run(now: datetime, at: str) -> datetime:
    """다음 실행 시각 (UTC 기준 매일 at=HH:MM)."""
    hour, minute = (int(x) for x in at.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def curve_points(log_dir: Path) -> dict:
    """rl-engine 학습 기록(TensorBoard)에서 스텝별 평균 에피소드 보상·길이를 읽는다."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    files = sorted(log_dir.glob("*/events.out.tfevents.*"))
    if not files:
        return {}
    events = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags()["scalars"])
    return {
        name: [(e.step, float(e.value)) for e in events.Scalars(tag)]
        for tag, name in (
            ("rollout/ep_rew_mean", "ep_rew_mean"),
            ("rollout/ep_len_mean", "ep_len_mean"),
        )
        if tag in tags
    }


def _clean(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value == value and abs(value) != float("inf"):  # NaN·무한대 제외
            out[key] = float(value)
    return out


def _production(client) -> tuple[object, dict]:
    version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION)
    return version, dict(version.tags)


def _reference_val_sharpe(client, window: int, variant: str, seed: str):
    found = client.get_experiment_by_name("rl-engine-val")
    if found is None:
        return None
    runs = client.search_runs(
        [found.experiment_id],
        f"tags.model_name = 'ppo_val_w{window}_{variant}_s{seed}'",
        max_results=1,
    )
    return runs[0].data.metrics.get("sharpe") if runs else None


def _already_done(client, fingerprint: str, config: str) -> bool:
    found = client.get_experiment_by_name(EXPERIMENT)
    if found is None:
        return False
    runs = client.search_runs(
        [found.experiment_id],
        f"tags.data_sha256 = '{fingerprint}' and tags.config = '{config}'",
        max_results=1,
    )
    return bool(runs)


def run_once(force: bool = False) -> dict:
    from mlflow import MlflowClient

    client = MlflowClient()
    production, tags = _production(client)
    variant, seed = tags["variant"], tags["seed"]
    reward_type, mdd_lambda = parse_variant(variant)
    lookback = LOOKBACK
    window = latest_window(RL_DATA_DIR)
    fingerprint = data_fingerprint(RL_DATA_DIR, window)
    config = f"w{window}_{variant}_s{seed}_t{TIMESTEPS}"
    if not force and _already_done(client, fingerprint, config):
        print(f"[retrain] 데이터가 바뀌지 않아 건너뜀 ({config}, {fingerprint[:12]})")
        return {"status": "skipped", "config": config}

    source = RL_REPO_PATH / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import train  # rl-engine 학습 코드 (수정하지 않고 호출만)
    from data_loader import load_walk_forward_window

    work = Path(tempfile.mkdtemp(prefix="retrain-"))
    # 저장 위치만 임시 폴더로 바꾼다 (rl-engine 폴더에 모델·로그를 남기지 않음)
    train.MODELS_DIR, train.LOGS_DIR = work / "models", work / "logs"
    data = load_walk_forward_window(window, RL_DATA_DIR)
    started = time.time()
    print(f"[retrain] 선택용 학습 시작: {config}")
    val = train.train_one(
        data,
        variant,
        reward_type,
        mdd_lambda,
        int(seed),
        TIMESTEPS,
        lookback,
        stage="val",
        data_dir=RL_DATA_DIR,
    )
    reference = _reference_val_sharpe(client, window, variant, seed)
    ok, reason = recommend(_clean(val).get("sharpe"), reference, MIN_GAIN)
    print(f"[retrain] 승격 판단: {'권장' if ok else '보류'} - {reason}")
    print("[retrain] 최종 학습 시작 (서빙용 test 단계 모델)")
    test = train.train_one(
        data,
        variant,
        reward_type,
        mdd_lambda,
        int(seed),
        TIMESTEPS,
        lookback,
        stage="test",
        data_dir=RL_DATA_DIR,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    base = f"ppo_test_w{window}_{variant}_s{seed}"
    name = f"{base}_r{stamp}"  # 원래 모델과 이름이 겹치지 않게
    checkpoint = work / "checkpoint"
    checkpoint.mkdir()
    shutil.copy(work / "models" / f"{base}.zip", checkpoint / f"{name}.zip")
    shutil.copy(work / "models" / f"{base}.json", checkpoint / f"{name}.json")

    found = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        found.experiment_id if found else client.create_experiment(EXPERIMENT)
    )
    run_tags = {
        "model_name": name,
        "variant": variant,
        "window": str(window),
        "seed": seed,
        "kind": "retrain",
        "config": config,
        "data_sha256": fingerprint,
        "promotion_recommended": str(ok).lower(),
        "promotion_reason": reason,
        "based_on_version": str(production.version),
    }
    run = client.create_run(experiment_id, run_name=name, tags=run_tags)
    for key, value in {
        "reward_type": reward_type,
        "mdd_lambda": mdd_lambda,
        "seed": seed,
        "timesteps": TIMESTEPS,
        "lookback": lookback,
        "window": window,
    }.items():
        client.log_param(run.info.run_id, key, value)
    for prefix, metrics in (("val", val), ("test", test)):
        for key, value in _clean(metrics).items():
            client.log_metric(run.info.run_id, f"{prefix}_{key}", value)
    # 학습 곡선 (최종 학습은 train/, 선택용 학습은 train_val/)
    for stage, prefix in (("test", "train"), ("val", "train_val")):
        log_dir = work / "logs" / f"ppo_{stage}_w{window}_{variant}_s{seed}"
        for key, points in curve_points(log_dir).items():
            for step, value in points:
                client.log_metric(run.info.run_id, f"{prefix}/{key}", value, step=step)
    if reference is not None:
        client.log_metric(run.info.run_id, "reference_val_sharpe", reference)
    client.log_metric(run.info.run_id, "train_seconds", time.time() - started)
    client.log_artifacts(run.info.run_id, str(checkpoint), "checkpoint")
    client.set_terminated(run.info.run_id)

    version = client.create_model_version(
        MODEL_NAME,
        source=f"{client.get_run(run.info.run_id).info.artifact_uri}/checkpoint",
        run_id=run.info.run_id,
        tags=run_tags,
        description=f"재학습 후보 ({reason})",
    )
    client.set_registered_model_alias(MODEL_NAME, CANDIDATE, version.version)
    shutil.rmtree(work, ignore_errors=True)
    print(
        f"[retrain] {MODEL_NAME}@{CANDIDATE} → v{version.version} ({name}), "
        f"승격 {'권장' if ok else '보류'}. 승격: python -m mlops.promote"
    )
    return {
        "status": "registered",
        "version": version.version,
        "recommended": ok,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="한 번 실행하고 끝냄")
    group.add_argument(
        "--schedule", action="store_true", help="매일 정해진 시각에 실행"
    )
    parser.add_argument("--force", action="store_true", help="데이터가 같아도 실행")
    args = parser.parse_args()
    if args.once:
        run_once(force=args.force)
        return
    while True:
        now = datetime.now(timezone.utc)
        target = next_run(now, RUN_AT)
        print(f"[retrain] 다음 실행: {target:%Y-%m-%d %H:%M} UTC", flush=True)
        time.sleep((target - now).total_seconds())
        try:
            run_once()
        except Exception as error:  # 하루 실패해도 스케줄러는 계속 돈다
            print(f"[retrain] 실패: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
