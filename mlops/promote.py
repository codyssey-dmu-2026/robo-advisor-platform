"""재학습 후보(candidate)를 운영(production)으로 승격하거나 직전 운영 버전으로 되돌린다.

자동 승격은 하지 않는다. 후보의 검증 성적·승격 권장 여부를 사람이 확인한 뒤 실행한다.

    python -m mlops.promote             # 운영·후보 버전 정보만 보여 줌
    python -m mlops.promote --yes       # 후보를 운영으로 승격
    python -m mlops.promote --rollback  # 직전 운영 버전으로 되돌림

승격 후 RL 서버를 다시 시작해야 새 버전을 서빙한다: docker compose restart rl-engine
"""

import argparse
import sys

from mlops.registry import CANDIDATE, MODEL_NAME, PRODUCTION

PREVIOUS = "previous_production"  # 되돌리기용으로 직전 운영 버전을 적어 두는 태그


def describe(version) -> str:
    tags = version.tags
    return (
        f"v{version.version} {tags.get('model_name')} "
        f"[{tags.get('kind', 'original')}] "
        f"승격 권장={tags.get('promotion_recommended', '-')} "
        f"({tags.get('promotion_reason', '원래 학습 모델')})"
    )


def main() -> None:
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--yes", action="store_true", help="후보를 운영으로 승격")
    action.add_argument("--rollback", action="store_true", help="직전 운영 버전으로")
    args = parser.parse_args()

    client = MlflowClient()
    production = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION)
    print(f"운영(production): {describe(production)}")
    try:
        candidate = client.get_model_version_by_alias(MODEL_NAME, CANDIDATE)
        print(f"후보(candidate):  {describe(candidate)}")
    except MlflowException:
        candidate = None
        print("후보(candidate):  없음 (python -m mlops.retrain --once로 만들 수 있음)")

    if args.rollback:
        previous = client.get_registered_model(MODEL_NAME).tags.get(PREVIOUS)
        if not previous:
            sys.exit("되돌릴 직전 운영 버전 기록이 없습니다")
        target = previous
    elif args.yes:
        if candidate is None:
            sys.exit("승격할 후보가 없습니다")
        if candidate.version == production.version:
            sys.exit("후보가 이미 운영 중입니다")
        target = candidate.version
    else:
        print("승격하려면 --yes, 되돌리려면 --rollback을 붙여 다시 실행하세요.")
        return

    client.set_registered_model_tag(MODEL_NAME, PREVIOUS, production.version)
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION, target)
    print(f"운영(production): v{production.version} → v{target}")
    print("RL 서버에 반영: docker compose restart rl-engine")


if __name__ == "__main__":
    main()
