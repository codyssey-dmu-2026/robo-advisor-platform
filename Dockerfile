# Gateway(8080)·RL 서빙(8002)·대시보드(8501)가 공유하는 이미지. 실행 명령은 docker-compose의 command로 지정
# 다른 팀 서비스(agentic-research, evaluation-xai)와 같은 Python 3.13 사용
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080 8002 8501
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8080"]
