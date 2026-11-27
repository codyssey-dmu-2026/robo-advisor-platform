"""MLflow 모델 저장소에서 서빙할 모델을 찾아 내려받는다.

모델 URI는 `models:/<등록 모델 이름>@<별칭>` 형식이다 (예: models:/ppo-portfolio@production).
등록할 때 모델 버전에 붙인 `model_name` 태그로 rl-engine 모델 이름(.zip·.json 파일명)을 찾는다.
mlflow는 RL 서빙 이미지에만 설치하므로 함수 안에서 불러온다.
"""

import re
from pathlib import Path

MODEL_NAME = "ppo-portfolio"  # MLflow에 등록하는 모델 이름
PRODUCTION = "production"  # 서빙 중인 버전에 붙이는 별칭
CANDIDATE = "candidate"  # 재학습으로 새로 만든 후보 버전에 붙이는 별칭
URI = re.compile(r"^models:/(?P<name>[\w.-]+)@(?P<alias>[\w-]+)$")


def parse_uri(uri: str) -> tuple[str, str]:
    match = URI.match(uri.strip())
    if not match:
        raise ValueError(f"모델 URI 형식이 다릅니다 (models:/<이름>@<별칭>): {uri}")
    return match["name"], match["alias"]


def resolve_model(uri: str, destination: Path) -> dict:
    """별칭이 가리키는 버전의 체크포인트(.zip·.json)를 destination에 내려받는다."""
    from mlflow import MlflowClient
    from mlflow.artifacts import download_artifacts

    name, alias = parse_uri(uri)
    version = MlflowClient().get_model_version_by_alias(name, alias)
    model_name = version.tags.get("model_name")
    if not model_name:
        raise ValueError(f"{name} v{version.version}에 model_name 태그가 없습니다")
    destination.mkdir(parents=True, exist_ok=True)
    local = Path(
        download_artifacts(artifact_uri=version.source, dst_path=str(destination))
    )
    if not (local / f"{model_name}.zip").is_file():
        raise FileNotFoundError(f"내려받은 폴더에 {model_name}.zip이 없습니다: {local}")
    return {
        "name": model_name,
        "dir": local,
        "source": "mlflow",
        "uri": uri,
        "registered_model": name,
        "alias": alias,
        "version": str(version.version),
        "run_id": version.run_id,
    }


def _version_summary(client, version) -> dict:
    tags = dict(version.tags)
    metrics = {}
    if version.run_id:
        run = client.get_run(version.run_id).data.metrics
        for key in ("val_sharpe", "reference_val_sharpe", "test_sharpe", "sharpe"):
            if key in run:
                metrics[key] = run[key]
    name = tags.get("model_name") or ""
    if "val_sharpe" not in metrics and name.startswith("ppo_test_"):
        # 원래 학습 모델은 같은 설정의 선택용(val) 학습 기록에서 검증 샤프를 가져온다
        found = client.get_experiment_by_name("rl-engine-val")
        if found is not None:
            runs = client.search_runs(
                [found.experiment_id],
                f"tags.model_name = '{name.replace('ppo_test_', 'ppo_val_', 1)}'",
                max_results=1,
            )
            if runs and "sharpe" in runs[0].data.metrics:
                metrics["val_sharpe"] = runs[0].data.metrics["sharpe"]
    return {
        "version": str(version.version),
        "model_name": tags.get("model_name"),
        "kind": tags.get("kind", "original"),
        "variant": tags.get("variant"),
        "seed": tags.get("seed"),
        "promotion_recommended": tags.get("promotion_recommended"),
        "promotion_reason": tags.get("promotion_reason"),
        "created_ms": version.creation_timestamp,
        "metrics": metrics,
    }


def summarize_aliases(name: str = MODEL_NAME) -> dict:
    """운영(production)·후보(candidate) 버전 요약과 전체 버전 수."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    model = client.get_registered_model(name)
    aliases = {}
    for alias in (PRODUCTION, CANDIDATE):
        try:
            version = client.get_model_version_by_alias(name, alias)
        except MlflowException:
            continue
        aliases[alias] = _version_summary(client, version)
    count = len(client.search_model_versions(f"name='{name}'"))
    return {"registered_model": name, "version_count": count, "aliases": aliases} | {
        "description": model.description
    }
