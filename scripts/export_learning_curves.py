"""rl-engine 학습 기록(TensorBoard)을 대시보드용 학습 곡선 CSV로 변환한다.

rl-engine은 학습 기록을 `logs/`에 남기지만 git에서는 제외한다. 팀원이 공유한 `logs` 폴더(또는 zip)를
이 스크립트로 한 번 변환해 `data/learning_curves.csv`로 저장하고, RL 서버가 그 파일을 제공한다.

    pip install tensorboard==2.20.0   # 변환할 때만 필요 (서버 이미지에는 넣지 않음)
    python scripts/export_learning_curves.py <logs 폴더 또는 zip> data/learning_curves.csv

- 실행 폴더 이름: ppo_{stage}_w{window}_{variant}_s{seed} (예: ppo_test_w4_simple_s42)
- 같은 실행을 다시 학습한 기록(PPO_1, PPO_2 …)이 있으면 번호가 가장 큰(마지막) 기록을 쓴다
- 100,000 스텝을 채우지 못한 기록은 빼고 이유를 출력한다
"""

import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

RUN = re.compile(
    r"^ppo_(?P<stage>test|val)_w(?P<window>\d+)_(?P<variant>.+)_s(?P<seed>\d+)$"
)
MIN_STEPS = 100_000  # 과제 최소 학습 스텝
TAGS = {
    "rollout/ep_rew_mean": "ep_rew_mean",
    "rollout/ep_len_mean": "ep_len_mean",
    "train/entropy_loss": "entropy_loss",
    "train/explained_variance": "explained_variance",
}


def read_events(path: Path) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    events = EventAccumulator(str(path), size_guidance={"scalars": 0})
    events.Reload()
    available = set(events.Tags()["scalars"])
    columns = {}
    for tag, name in TAGS.items():
        if tag in available:
            columns[name] = pd.Series(
                {e.step: e.value for e in events.Scalars(tag)}, dtype=float
            )
    return pd.DataFrame(columns).rename_axis("step").reset_index()


def latest_attempt(run_dir: Path) -> tuple[str, Path] | None:
    """PPO_n 폴더 중 번호가 가장 큰 것의 이벤트 파일."""
    attempts = sorted(
        (int(d.name.split("_")[-1]), d)
        for d in run_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"PPO_\d+", d.name)
    )
    if not attempts:
        return None
    _, folder = attempts[-1]
    files = sorted(folder.glob("events.out.tfevents.*"))
    return (folder.name, files[-1]) if files else None


def export(logs_dir: Path) -> pd.DataFrame:
    frames, skipped = [], []
    for run_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        match = RUN.match(run_dir.name)
        found = latest_attempt(run_dir) if match else None
        if found is None:
            skipped.append((run_dir.name, "형식이 다르거나 기록 없음"))
            continue
        attempt, path = found
        curve = read_events(path)
        if curve.empty or curve["step"].max() < MIN_STEPS:
            skipped.append((run_dir.name, f"{curve['step'].max()} 스텝에서 끝남"))
            continue
        frames.append(
            curve.assign(
                run=run_dir.name,
                stage=match["stage"],
                window=int(match["window"]),
                variant=match["variant"],
                seed=int(match["seed"]),
                attempt=attempt,
            )
        )
    for name, reason in skipped:
        print(f"[건너뜀] {name}: {reason}")
    columns = ["run", "stage", "window", "variant", "seed", "attempt", "step"]
    result = pd.concat(frames, ignore_index=True)
    return result[columns + [c for c in TAGS.values() if c in result]]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as tmp:
        if source.suffix == ".zip":
            with zipfile.ZipFile(source) as archive:
                archive.extractall(tmp)
            source = next(Path(tmp).rglob("logs"), Path(tmp))
        curves = export(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output, index=False, float_format="%.6g")
    runs = curves.groupby(["stage"])["run"].nunique().to_dict()
    print(f"{output}: 실행 {curves['run'].nunique()}개 {runs}, {len(curves)}행")


if __name__ == "__main__":
    main()
