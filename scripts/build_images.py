"""docker-compose 이미지 빌드 헬퍼 (Windows용 우회).

Windows의 Docker Compose 5.1.x는 GitHub URL build context를 bake에 넘길 때
URL을 로컬 경로로 해석해 실패한다(`failed to evaluate path "https://..."`).
`docker compose build --print`로 뽑은 빌드 정의를 `docker buildx bake`에 직접 넘기면
정상 동작하므로 그 과정을 대신 수행한다. 바이트를 그대로 전달해 한글 경로도 깨지지 않는다.

사용법 (레포 루트에서):
    python scripts/build_images.py                 # 기본 모드(GitHub 고정 커밋)
    python scripts/build_images.py --local         # docker-compose.local.yml 포함
    python scripts/build_images.py api-gateway     # 특정 서비스만
빌드 후 `docker compose up` (로컬 모드는 -f 두 개)으로 실행한다.
Linux/macOS에서는 이 스크립트 없이 `docker compose up --build`를 쓰면 된다.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("services", nargs="*", help="빌드할 서비스 (기본: 전체)")
    parser.add_argument("--local", action="store_true", help="docker-compose.local.yml 포함")
    args = parser.parse_args()

    compose = ["docker", "compose", "-f", "docker-compose.yml"]
    if args.local:
        compose += ["-f", "docker-compose.local.yml"]

    definition = subprocess.run(
        [*compose, "build", "--print", *args.services],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if definition.returncode != 0:
        sys.stderr.buffer.write(definition.stderr)
        return definition.returncode

    bake = subprocess.run(
        ["docker", "buildx", "bake", "-f", "-"], cwd=ROOT, input=definition.stdout, check=False
    )
    return bake.returncode


if __name__ == "__main__":
    sys.exit(main())
