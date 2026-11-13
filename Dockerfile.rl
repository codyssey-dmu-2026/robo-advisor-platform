# syntax=docker/dockerfile:1
# RL 서빙 서버(8002) 전용 이미지. torch 용량이 커서 Gateway·대시보드 이미지와 분리
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements-rl.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install --no-cache-dir -r requirements-rl.txt

# rl-engine 환경 코드 (커밋 고정)
ADD https://github.com/codyssey-dmu-2026/rl-engine.git#05579d2c845e959fd7948f8cd5a826ca0a3b0599 /opt/rl-engine

# 학습 완료 모델 번들 (Release models-v1, 지문 검사). 전체를 넣어 RL_MODEL_NAME만으로 교체 가능
ADD --checksum=sha256:d4b5df070fdcfd534cf08fec76e94b1aab5a3e9e7e0a6a9ca125132c37753e5e \
    https://github.com/codyssey-dmu-2026/robo-advisor-platform/releases/download/models-v1/rl-engine_test_models.zip \
    /tmp/models.zip
RUN python -m zipfile -e /tmp/models.zip /opt/rl-models && rm /tmp/models.zip

COPY rl_server.py .
ENV RL_REPO_PATH=/opt/rl-engine RL_MODELS_DIR=/opt/rl-models/models
EXPOSE 8002
CMD ["uvicorn", "rl_server:app", "--host", "0.0.0.0", "--port", "8002"]
