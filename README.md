# Robo-Advisor Platform (API Gateway, Web Dashboard & MLOps)

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
    ├── rl_server.py         # PPO 강화학습 모델 추론 및 Softmax 비중 서빙 서버 (Port 8002)
    ├── dashboard/           # Streamlit 웹 대시보드 (Port 8501)
    │   ├── app.py           # 진입점: 사이드바 내비게이션·시스템 상태·면책 조항
    │   ├── views/           # 개요 + 6개 필수 페이지
    │   ├── core/            # Gateway HTTP 클라이언트, 차트, 공통 UI, 자산 정보
    │   └── static/          # 로고
    ├── .streamlit/config.toml  # 라이트/다크 테마 (색각 이상 대비 검증 팔레트)
    ├── tests/
    │   ├── test_api.py      # 5대 엔드포인트 규격, 422 방어, 5초 SLA 검증 (14개)
    │   └── test_dashboard.py  # 시계열 API, Gateway 오프라인 시 페이지 렌더링 검증 (9개)
    ├── scripts/
    │   └── build_images.py  # Windows용 이미지 빌드 헬퍼 (Compose git URL 빌드 버그 우회)
    ├── Dockerfile           # API 서빙 및 대시보드 컨테이너 이미지 빌드 명세
    ├── docker-compose.yml   # 전체 통합 실행 (팀원 레포 커밋 고정 빌드 + bootstrap)
    ├── docker-compose.local.yml  # 개발용: 형제 폴더(../) 코드로 빌드
    ├── .env.example         # 설정·선택 키 템플릿
    ├── requirements.txt     # 의존성 패키지 목록
    └── README.md

---

## 🔌 5대 필수 API 엔드포인트 명세 (Gateway `8080`)

Swagger UI 문서: 서버 실행 후 `http://127.0.0.1:8080/docs` 접속

| 메서드 | 엔드포인트 | 설명 | 연동 마이크로서비스 |
|---|---|---|---|
| `GET` | `/health` | 전체 게이트웨이 및 하위 3개 서비스 상태/모델 로드 여부 반환 | `8000`, `8001`, `8002` |
| `POST` | `/optimize` | 12개 자산 유니버스 기반 최적 포트폴리오 비중(합=1.0) 반환 | `rl-engine (8002)` |
| `POST` | `/explain` | 특정 의사결정(`run_id`, `decision_id`, `target_asset`) SHAP 기여도 반환 | `evaluation-xai (8000)` |
| `POST` | `/research` | 투자 질문에 대한 LangGraph 리서치 요약, 추론 로그, 출처, RiskTag 반환 | `agentic-research (8001)` |
| `GET` | `/backtest` | Walk-Forward 백테스트 12개 성과 지표 및 ANOVA/Bootstrap 통계 결과 반환 | `evaluation-xai (8000)` |
| `GET` | `/backtest/{run_id}/timeseries` | (보조) 일별 누적 수익률·낙폭·비중 시계열. 대시보드 차트용 | `evaluation-xai` 산출물 읽기 전용 |

---

## 🖥️ Streamlit 대시보드 (`8501`)

사이드바 내비게이션, 서비스 상태 표시, 라이트/다크 테마(우측 상단 ⋮ 메뉴의 System·Light·Dark), 모든 차트에 '표로 보기' 제공.

0. **개요:** 핵심 KPI(누적 수익률·샤프·MDD·VaR, 추세선 포함), 전략별 누적 수익률, 자산군 비중
1. **포트폴리오 현황 (`POST /optimize`):** 종목별 비중 막대, 자산군 도넛, 전체 테스트 구간 누적 수익률, 비중 변화 영역 차트
2. **강화학습 성과 (`GET /backtest`):** 지표 선택형 전략×구간 비교 막대, 12개 지표 전체 표, 학습 곡선(rl-engine 결과 연동 예정)
3. **SHAP 해석 (`POST /explain`):** 기준 비중 → 피처 그룹별 기여 → 예측 비중 폭포 차트, 해석 문장
4. **에이전트 리서치 (`POST /research`):** 예시 질문, 투자의견 배지·신뢰도, 단계별 추론 과정(Reasoning Trace), 인용 번호가 달린 주장, 원문 링크, 리스크 태그
5. **ANOVA 검증 결과 (`GET /backtest`):** 과제 3대 검증 상태, 효과 검정(F·p·η²)과 유의성, 가정 진단, 블록 부트스트랩 신뢰구간
6. **리스크 모니터링:** VaR·CVaR·MDD 게이지, 낙폭(Underwater) 차트와 -15% 한도선, Safe-Guard 발동 기록 및 상태 알림

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

### C. 개발 모드 (형제 폴더의 작업 중 코드로 빌드)

상위 폴더에 `data-pipeline`, `evaluation-xai`, `agentic-research`, `rl-engine`, `robo-advisor-platform`이 나란히 있을 때:

    python scripts/build_images.py --local
    docker compose -f docker-compose.yml -f docker-compose.local.yml up

---

## 🧪 유닛 테스트 실행 (`pytest`)

5대 엔드포인트 등록 여부, 스키마 검증(422 방어), 비중 합 1.0 제약, 5초 이내 응답(SLA)을 검증하는 12개의 유닛 테스트를 포함합니다.

    python -m pytest -v

---

> ⚠️ **[금융 면책 조항]** 본 시스템은 대학교 캡스톤 디자인 교육 목적으로 개발되었으며, 실제 투자 조언이나 자문에 사용할 수 없습니다. 백테스팅 및 시뮬레이션 성과가 미래의 수익을 보장하지 않습니다.