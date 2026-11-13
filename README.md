# Robo-Advisor Platform (API Gateway, Web Dashboard & MLOps)

[![Platform checks](https://github.com/codyssey-dmu-2026/robo-advisor-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/codyssey-dmu-2026/robo-advisor-platform/actions/workflows/ci.yml)

에이전틱 RAG + 강화학습 기반 통합 자율 로보어드바이저 프로젝트의 **API 서빙, 웹 대시보드 및 인프라(MLOps)** 파트 레포지토리입니다.

프론트엔드(Streamlit)에서 모델을 직접 로드하지 않는 **MSA(Microservice Architecture) 원칙**을 준수하며, 하위 마이크로서비스(`evaluation-xai`, `agentic-research`, `rl-engine`)를 통합하는 단일 진입점 **API Gateway (`8080`)**와 **6개 필수 탭 기반 대시보드 (`8501`)**를 제공합니다.

---

## 🏛️ 시스템 아키텍처 및 포트 구성

    [ 사용자 브라우저 ]
           │
           ▼ (Port 8501)
    ┌─────────────────────────────────────────────────────────┐
    │  Streamlit Web Dashboard (dashboard/app.py)             │
    │  - 개요 + 6개 필수 페이지, 라이트/다크 테마 (Plotly)    │
    └─────────────────────────────────────────────────────────┘
           │ (HTTP REST API 통신 - 모델 직접 로드 금지 준수)
           ▼ (Port 8080)
    ┌─────────────────────────────────────────────────────────┐
    │  FastAPI Unified API Gateway (gateway.py)               │
    │  - 5대 엔드포인트: /health, /optimize, /explain,        │
    │                    /research, /backtest                 │
    └─────────┬───────────────────┬───────────────────┬───────┘
              │                   │                   │
              ▼ (Port 8000)       ▼ (Port 8001)       ▼ (Port 8002)
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │  evaluation-xai  │ │ agentic-research │ │    rl-engine     │
    │ (Walk-Forward,   │ │ (LangGraph RAG,  │ │ (PPO Softmax,    │
    │  SHAP, ANOVA)    │ │  Chroma, RiskTag)│ │  MDD Safe-Guard) │
    └──────────────────┘ └──────────────────┘ └──────────────────┘

---

## 📂 폴더 구조

    robo-advisor-platform/
    ├── gateway.py           # 5대 필수 엔드포인트 통합 FastAPI 게이트웨이 (Port 8080)
    ├── rl_server.py         # rl-engine PPO 모델 추론·비중 서빙 서버 (Port 8002)
    ├── dashboard/           # Streamlit 웹 대시보드 (Port 8501)
    │   ├── app.py           # 진입점: 사이드바 내비게이션·시스템 상태·면책 조항
    │   ├── views/           # 개요 + 6개 필수 페이지
    │   ├── core/            # Gateway HTTP 클라이언트, 차트, 공통 UI, 자산 정보
    │   └── static/          # 로고
    ├── .streamlit/config.toml  # 라이트/다크 테마 (색각 이상 대비 검증 팔레트)
    ├── tests/
    │   ├── test_api.py      # 5대 엔드포인트 규격, RL 서빙 규칙, 422 방어, 5초 SLA 검증
    │   ├── test_performance.py  # 실제 백테스트 성과 API 검증
    │   ├── test_dashboard.py  # 시계열 API, 실데이터·오프라인 페이지 렌더링 검증
    │   └── conftest.py      # 테스트용 백테스트 결과 CSV 준비
    ├── scripts/
    │   └── build_images.py  # Windows용 이미지 빌드 헬퍼 (Compose git URL 빌드 버그 우회)
    ├── Dockerfile           # Gateway·대시보드 컨테이너 이미지 빌드 명세
    ├── Dockerfile.rl        # RL 서빙 전용 이미지 (torch CPU, 모델 번들·rl-engine 코드 고정)
    ├── requirements-rl.txt  # RL 서빙 의존성 (버전 고정)
    ├── docker-compose.yml   # 전체 통합 실행 (팀원 레포 커밋 고정 빌드 + bootstrap)
    ├── docker-compose.local.yml  # 개발용: 형제 폴더(../) 코드로 빌드
    ├── .env.example         # 설정·선택 키 템플릿
    ├── requirements.txt     # 의존성 패키지 목록 (버전 고정)
    ├── requirements-dev.txt # 개발용: black, flake8
    ├── pyproject.toml, .flake8  # 코드 스타일 설정 (한 줄 88자)
    └── README.md

---

## 🔌 5대 필수 API 엔드포인트 명세 (Gateway `8080`)

Swagger UI 문서: 서버 실행 후 `http://127.0.0.1:8080/docs` 접속 (요청·응답 스키마 모두 문서화)

- 요청 형식이 잘못되면 하위 서비스에 보내기 전에 `422`로 거절합니다 (팀원 서비스와 같은 규칙).
- 하위 서비스가 꺼져 있으면 `503`, 제한 시간(요청 1건 5초 기준) 안에 응답하지 않으면 `504`, 그 밖의 오류는 하위 서비스의 상태 코드와 사유를 그대로 전달합니다.
- `/optimize`만 예외로, RL 서버 장애 시 `status: mock_fallback`(동일 비중)으로 응답합니다.

| 메서드 | 엔드포인트 | 설명 | 연동 마이크로서비스 |
|---|---|---|---|
| `GET` | `/health` | 전체 게이트웨이 및 하위 3개 서비스 상태/모델 로드 여부 반환 | `8000`, `8001`, `8002` |
| `POST` | `/optimize` | 12개 자산 유니버스 기반 최적 포트폴리오 비중(합=1.0) 반환 | `rl-engine (8002)` |
| `POST` | `/explain` | 특정 의사결정(`run_id`, `decision_id`, `target_asset`) SHAP 기여도 반환 | `evaluation-xai (8000)` |
| `POST` | `/research` | 투자 질문에 대한 LangGraph 리서치 요약, 추론 로그, 출처, RiskTag 반환. `tickers`·`as_of`를 생략하면 에이전트가 질문에서 종목을 인식하고 요청 시각을 기준으로 함 | `agentic-research (8001)` |
| `GET` | `/backtest` | Walk-Forward 백테스트 12개 성과 지표 및 ANOVA/Bootstrap 통계 결과 반환 | `evaluation-xai (8000)` |
| `GET` | `/backtest/{run_id}/timeseries` | (보조) 일별 누적 수익률·낙폭·비중 시계열. 대시보드 차트용 | `evaluation-xai` 산출물 읽기 전용 |
| `GET` | `/performance` | (보조) 현재 RL 모델의 Walk-Forward 실제 성과: 12개 지표, 동일가중·SPY·보상 함수 5종 비교, 4년 연결 시계열, Safe-Guard 한도 초과 구간 | `rl-engine (8002)` |

---

## 🖥️ Streamlit 대시보드 (`8501`)

사이드바 내비게이션, 서비스 상태 표시, 라이트/다크 테마(우측 상단 ⋮ 메뉴의 System·Light·Dark), 모든 차트에 '표로 보기' 제공.

0. **개요:** 현재 모델의 4년 KPI(누적 수익률·샤프·MDD·VaR, 동일가중 대비 차이), 현재 모델·동일가중·SPY 누적 수익률, 자산군 비중
1. **포트폴리오 현황 (`POST /optimize`, `GET /performance`):** 실제 PPO 모델의 기준일별 종목 비중, 자산군 도넛, 기준일 낙폭 기반 Safe-Guard 상태, 4년 연결 누적 수익률, 비중 변화
2. **강화학습 성과 (`GET /performance`):** 과제 성능 기준 점검(SPY·동일가중 대비 샤프 +0.2, 누적 +10%p), 보상 함수 5종·벤치마크 연도별 비교, 12개 지표 전체 표, 학습 곡선(rl-engine 학습 기록 공유 대기)
3. **SHAP 해석 (`POST /explain`):** 기준 비중 → 피처 그룹별 기여 → 예측 비중 폭포 차트, 해석 문장
4. **에이전트 리서치 (`POST /research`):** 예시 질문, 투자의견 배지·신뢰도, 단계별 추론 과정(Reasoning Trace), 인용 번호가 달린 주장, 원문 링크, 리스크 태그
5. **ANOVA 검증 결과 (`GET /backtest`):** 과제 3대 검증 상태, 효과 검정(F·p·η²)과 유의성, 가정 진단, 블록 부트스트랩 신뢰구간
6. **리스크 모니터링 (`GET /performance`):** 4년 전체·연도별 VaR·CVaR·MDD 게이지, 낙폭(Underwater) 차트와 -15% 한도선, Safe-Guard 한도 초과 구간 기록 및 상태 알림

SHAP·ANOVA 페이지는 evaluation-xai 결과를 사용하며, 실데이터 평가 전까지 `DEMO` 배지로 표시됩니다.

---

## 🚀 실행 방법

### A. 로컬 환경 실행

    # 1. 가상환경 생성 및 패키지 설치
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

    # 2. RL 서빙 서버 (8002) 및 API Gateway (8080) 실행
    python -m uvicorn rl_server:app --host 127.0.0.1 --port 8002
    python -m uvicorn gateway:app --host 127.0.0.1 --port 8080

    # 3. Streamlit 대시보드 (8501) 실행
    python -m streamlit run dashboard/app.py

### B. Docker Compose 일괄 실행 (권장)

이 레포만 clone하면 됩니다. 팀원 서비스는 GitHub 레포를 **커밋 고정**으로 빌드합니다.

    cp .env.example .env        # 선택: DART 키·SEC 연락처·OpenAI 키 (비워 두면 키 없이 동작)
    docker compose up --build   # Linux / macOS

Windows의 Docker Compose 5.1.x는 GitHub URL 빌드에 버그가 있어 헬퍼로 먼저 빌드합니다.

    python scripts/build_images.py
    docker compose up

첫 실행 시 bootstrap 컨테이너가 팀원 코드로 데이터를 직접 만들고(수 분 소요), 두 번째부터는 건너뜁니다.

| 순서 | 컨테이너 | 하는 일 |
|---|---|---|
| 1 | `data-bootstrap` | data-pipeline: 가격 수집(yfinance/pykrx) → 전처리 → 뉴스·공시 수집(SEC, 한국경제 RSS, DART는 키 있을 때) |
| 2 | `agent-bootstrap` | agentic-research: 뉴스 → ChromaDB 적재, RiskTag 생성 |
| 3 | `eval-bootstrap` | evaluation-xai: 평가 산출물 생성 (현재 synthetic 데모) |

- 뉴스 데이터는 재배포 금지라 git에 올리지 않고 실행 환경마다 직접 수집합니다.
- 다시 수집하려면 `.env`에 `BOOTSTRAP_FORCE=1`, 완전 초기화는 `docker compose down -v`.

### RL 서빙 모델

- `rl-engine`에서 학습한 PPO 모델 번들을 Release [`models-v1`](https://github.com/codyssey-dmu-2026/robo-advisor-platform/releases/tag/models-v1)에 두고, `Dockerfile.rl`이 SHA-256 지문을 확인한 뒤 이미지에 넣습니다.
- 사용할 모델은 `.env`의 `RL_MODEL_NAME`으로 정합니다(기본 `ppo_test_w4_simple_s42`: 2025년 시험 구간, 검증 성적 기준 보상 방식). 값을 바꾸고 `docker compose up -d`만 다시 하면 교체됩니다.
- 서버는 시작할 때 시험 구간 전체를 모델로 재생해 날짜별 비중을 계산하므로 rl-engine 백테스트와 같은 비중이 나옵니다. `as_of`는 그 시각 **전에 끝난 거래일**의 결정만 사용합니다(미래 정보 누수 방지).
- 모델이 학습한 12개 자산 전체만 지원하며, 범위 밖 기준일·일부 자산 요청은 422로 거절합니다. 모델을 쓸 수 없으면 RL 서버는 503을, Gateway는 `status: mock_fallback`(동일 비중)을 반환합니다.
- 로컬 실행 시 `.env.example`의 `RL_REPO_PATH`·`RL_MODELS_DIR`·`RL_DATA_DIR`를 지정하고 `requirements-rl.txt`를 설치합니다.

### C. 개발 모드 (형제 폴더의 작업 중 코드로 빌드)

상위 폴더에 `data-pipeline`, `evaluation-xai`, `agentic-research`, `rl-engine`, `robo-advisor-platform`이 나란히 있을 때:

    python scripts/build_images.py --local
    docker compose -f docker-compose.yml -f docker-compose.local.yml up

---

## 🧪 테스트와 코드 스타일

5대 엔드포인트 등록 여부, 스키마 검증(422 방어), 비중 합 1.0 제약, 5초 이내 응답(SLA), 기준일 미래 정보 차단,
RL 장애 시 대체 응답, 실제 성과 API(4년 연결·시드 평균·벤치마크 지표·Safe-Guard 구간), 시계열 API,
실데이터·Gateway 오프라인 각각에서 대시보드 페이지 렌더링을 검증하는 53개의 테스트를 포함합니다. 하위 서비스는 가짜 응답으로 대체해 정상 전달·422·503·504를 각각 확인합니다.
(CI에는 torch가 없으므로 RL 서버 테스트는 가짜 비중 표를 넣어 API 규칙을 검증합니다.)
push·PR마다 GitHub Actions(`.github/workflows/ci.yml`)가 black·flake8·pytest와 Docker 이미지 빌드를 자동 실행합니다.

    pip install -r requirements-dev.txt   # pytest + black + flake8 (팀원 레포와 같은 버전)
    python -m pytest -v

코드 스타일은 팀 공통 규칙(한 줄 88자)으로 black 포매터와 flake8 린터를 적용합니다. 설정은 `pyproject.toml`, `.flake8`.

    python -m black .        # 형식 자동 정리
    python -m flake8 .       # 남은 문제 검사 (0건이어야 함)

---

> ⚠️ **[금융 면책 조항]** 본 시스템은 대학교 캡스톤 디자인 교육 목적으로 개발되었으며, 실제 투자 조언이나 자문에 사용할 수 없습니다. 백테스팅 및 시뮬레이션 성과가 미래의 수익을 보장하지 않습니다.