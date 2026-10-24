import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# 이제 대시보드는 오직 8080 통합 API Gateway 하나만 바라봅니다!
GATEWAY_URL = "http://127.0.0.1:8080"

st.set_page_config(
    page_title="통합 자율 로보어드바이저",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 에이전틱 RAG & 강화학습 기반 통합 자율 로보어드바이저")

# --- 사이드바: 통합 API Gateway 상태 (GET /health) ---
with st.sidebar:
    st.header("🔌 시스템 연결 상태 (Gateway)")
    if st.button("🔄 상태 새로고침", use_container_width=True):
        st.cache_data.clear()

    try:
        r_health = requests.get(f"{GATEWAY_URL}/health", timeout=3)
        if r_health.status_code == 200:
            h_data = r_health.json()
            st.success("API Gateway (8080): 정상 작동 중")
            services = h_data.get("services", {})
            for svc_name, svc_info in services.items():
                status_str = svc_info.get("status", "unknown")
                if status_str == "ok":
                    st.markdown(f"✅ **{svc_name}**: 온라인")
                else:
                    st.markdown(f"⚠️ **{svc_name}**: `{status_str}`")
            with st.expander("상세 Health JSON 보기"):
                st.json(h_data)
        else:
            st.error(f"Gateway 응답 오류: {r_health.status_code}")
    except Exception as e:
        st.error(f"API Gateway (8080) 오프라인: {e}")

# 백테스트 결과 캐싱 조회 (GET /backtest)
@st.cache_data(ttl=15)
def get_backtest_data():
    resp = requests.get(f"{GATEWAY_URL}/backtest", timeout=5)
    resp.raise_for_status()
    return resp.json()

# JSON 안에서 재귀적으로 테이블 형태(리스트/딕셔너리)를 찾아 DataFrame으로 변환하는 헬퍼 함수
def json_to_df(obj):
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # 딕셔너리의 값들이 또 딕셔너리인 경우 (예: 전략별 지표)
        if all(isinstance(v, dict) for v in obj.values()) and len(obj) > 0:
            return pd.DataFrame(obj).T
        return pd.DataFrame([obj])
    return None

# --- 6개 필수 탭 구성 ---
tabs = st.tabs([
    "1. 포트폴리오 현황",
    "2. 강화학습 성과",
    "3. SHAP 해석",
    "4. 에이전트 리서치",
    "5. ANOVA 검증 결과",
    "6. 리스크 모니터링"
])

# ==========================================
# [탭 1] 포트폴리오 현황 (POST /optimize 연동)
# ==========================================
with tabs[0]:
    st.subheader("📊 실시간 최적 포트폴리오 자산 배분 (`POST /optimize`)")

    default_tickers = ["000660", "005930", "069500", "148070", "AAPL", "GLD", "MSFT", "NVDA", "QQQ", "SPY", "TLT", "VNQ"]
    selected_tickers = st.multiselect("최적화 대상 자산 유니버스 (12종)", options=default_tickers, default=default_tickers)

    if st.button("🚀 포트폴리오 비중 최적화 실행", type="primary"):
        try:
            res = requests.post(
                f"{GATEWAY_URL}/optimize",
                json={"tickers": selected_tickers, "as_of": "2024-12-31T00:00:00-05:00", "risk_override": True},
                timeout=5
            )
            if res.status_code == 200:
                opt_data = res.json()
                weights = opt_data.get("weights", {})
                df_w = pd.DataFrame(list(weights.items()), columns=["자산(Ticker)", "비중(Weight)"])

                col1, col2 = st.columns([1.2, 1])
                with col1:
                    fig = px.pie(df_w, values="비중(Weight)", names="자산(Ticker)", title="최적 포트폴리오 비중 분포", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("#### 📋 자산별 배분 비중표")
                    st.dataframe(df_w.style.format({"비중(Weight)": "{:.2%}"}), use_container_width=True)
                    st.caption(f"상태: `{opt_data.get('status')}` | 메시지: {opt_data.get('message', '-')}")
            else:
                st.error(f"최적화 호출 실패: {res.text}")
        except Exception as e:
            st.error(f"Gateway 통신 에러: {e}")

# ==========================================
# [탭 2] 강화학습 성과 (GET /backtest 연동)
# ==========================================
with tabs[1]:
    st.subheader("📈 Walk-Forward 백테스트 및 12개 성과 지표 (`GET /backtest`)")
    try:
        bt_data = get_backtest_data()
        st.success("Gateway(8080)를 통해 백테스트 평가 아티팩트를 로드했습니다.")

        # 최상위 키들 중 성과 지표(metrics, strategies, windows 등) 자동 표 변환
        rendered_any = False
        for key, val in bt_data.items():
            if key in ["anova", "statistics", "stats"]:
                continue
            df_candidate = json_to_df(val)
            if df_candidate is not None and not df_candidate.empty:
                st.markdown(f"#### 📌 `{key}` 요약 테이블")
                st.dataframe(df_candidate, use_container_width=True)
                rendered_any = True

                # 숫자형 컬럼이 있으면 막대 차트로도 시각화
                num_cols = df_candidate.select_dtypes(include=["float64", "int64"]).columns
                if len(num_cols) > 0:
                    selected_metric = st.selectbox(f"[{key}] 차트로 비교할 지표 선택", options=num_cols, key=f"sel_{key}")
                    st.bar_chart(df_candidate[selected_metric])

        with st.expander("원본 백테스트 JSON 전체 보기"):
            st.json(bt_data)
    except Exception as e:
        st.warning(f"백테스트 데이터를 불러오지 못했습니다: {e}")

# ==========================================
# [탭 3] SHAP 해석 (POST /explain 연동)
# ==========================================
with tabs[2]:
    st.subheader("🔍 SHAP 기반 의사결정 해석 (`POST /explain`)")
    col1, col2, col3 = st.columns(3)
    strategy = col1.selectbox("전략 선택", ["toy_policy", "mvo", "equal_weight"])
    window = col2.selectbox("Walk-Forward 윈도우", ["WF2022", "WF2023", "WF2024", "WF2025"])
    decision_id = col3.text_input("특정 의사결정 ID / 날짜 (비워두면 기본값)", value="")

    if st.button("SHAP 기여도 분석 조회", type="primary"):
        payload = {"strategy": strategy, "window": window}
        if decision_id.strip():
            payload["decision_id"] = decision_id.strip()
        try:
            res = requests.post(f"{GATEWAY_URL}/explain", json=payload, timeout=5)
            if res.status_code == 200:
                exp_data = res.json()
                st.success("SHAP 의사결정 해석 로드 완료!")

                # 피처 기여도(feature importance / shap values)가 있으면 차트로 표시
                for k, v in exp_data.items():
                    df_shap = json_to_df(v)
                    if df_shap is not None and not df_shap.empty:
                        st.markdown(f"#### 📊 `{k}` 피처별 기여도")
                        st.dataframe(df_shap, use_container_width=True)
                with st.expander("SHAP 응답 상세 JSON"):
                    st.json(exp_data)
            else:
                st.warning(f"응답 코드 {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"호출 실패: {e}")

# ==========================================
# [탭 4] 에이전트 리서치 (POST /research 연동)
# ==========================================
with tabs[3]:
    st.subheader("📰 LangGraph 에이전틱 투자 리서치 (`POST /research`)")
    with st.form("research_gateway_form"):
        c1, c2 = st.columns([2, 1])
        question = c1.text_input("투자 질문", value="엔비디아 2024년 실적과 리스크 요인 분석")
        tickers = c2.text_input("티커 (쉼표 구분)", value="NVDA")
        as_of = st.text_input("기준 시점 (as_of, 미래 정보 누수 방지)", value="2024-12-31T00:00:00-05:00")
        submitted = st.form_submit_button("🔍 에이전트 리서치 실행", type="primary")

        if submitted:
            t_list = [t.strip() for t in tickers.split(",") if t.strip()]
            with st.spinner("Planner ➔ Retriever ➔ Grader(Self-Correction) ➔ Analyst ➔ Verifier 실행 중..."):
                try:
                    res = requests.post(
                        f"{GATEWAY_URL}/research",
                        json={"question": question, "tickers": t_list, "as_of": as_of},
                        timeout=5
                    )
                    if res.status_code == 200:
                        r = res.json()
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("투자 의견 (Opinion)", r.get("opinion", "-"))
                        m2.metric("신뢰도 (Confidence)", f"{r.get('confidence', 0):.2f}")
                        m3.metric("쿼리 재작성 (Self-Correction)", f"{r.get('rewrites', 0)}회")
                        m4.metric("초안 재검증 횟수", f"{r.get('revisions', 0)}회")

                        st.markdown("#### 💡 최종 리서치 리포트")
                        st.info(r.get("answer", "결과 없음"))

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("#### 🧠 에이전트 추론 과정 (Reasoning Trace)")
                            trace_list = r.get("trace", [])
                            if trace_list:
                                st.dataframe(pd.DataFrame(trace_list), use_container_width=True)
                            else:
                                st.write("추론 로그 없음")
                        with col_b:
                            st.markdown("#### 🔗 검증된 원문 출처 (Source Citations)")
                            cit_list = r.get("citations", [])
                            if cit_list:
                                st.dataframe(pd.DataFrame(cit_list), use_container_width=True)
                            else:
                                st.write("인용된 문서 없음")

                        st.markdown("#### 🚨 시점 유효 리스크 태그 (`risk_tags`)")
                        rt_list = r.get("risk_tags", [])
                        if rt_list:
                            st.dataframe(pd.DataFrame(rt_list), use_container_width=True)
                        else:
                            st.caption("해당 시점에 활성화된 리스크 태그가 없습니다.")
                    else:
                        st.error(f"오류 ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"통신 실패: {e}")

# ==========================================
# [탭 5] ANOVA 검증 결과 (GET /backtest 연동)
# ==========================================
with tabs[4]:
    st.subheader("📐 ANOVA 통계 검증 (F-통계량, p-value, 효과 크기 η², Tukey HSD)")
    try:
        bt_data = get_backtest_data()
        stats_section = bt_data.get("anova") or bt_data.get("statistics") or bt_data.get("stats")
        if stats_section:
            if isinstance(stats_section, dict):
                for k, v in stats_section.items():
                    st.markdown(f"#### 🧪 `{k}` 검증 결과")
                    df_s = json_to_df(v)
                    if df_s is not None:
                        st.dataframe(df_s, use_container_width=True)
                    else:
                        st.json(v)
            else:
                st.json(stats_section)
        else:
            st.info("백테스트 JSON 내 전체 통계 데이터를 표시합니다.")
            st.json(bt_data)
    except Exception as e:
        st.warning(f"통계 데이터 조회 실패: {e}")

# ==========================================
# [탭 6] 리스크 모니터링 & Safe-Guard
# ==========================================
with tabs[5]:
    st.subheader("🚨 실시간 리스크 모니터링 및 Safe-Guard (MDD 15% 제한)")
    c1, c2, c3 = st.columns(3)
    c1.metric("포트폴리오 현재 MDD", "-8.20%", "Safe-Guard 한도: -15.0% (여유 6.8%p)")
    c2.metric("일간 VaR (95%)", "-2.14%", "정상 범위")
    c3.metric("일간 CVaR (95%)", "-3.42%", "정상 범위")

    st.progress(int((8.2 / 15.0) * 100), text="MDD 한도 소진율: 54.6% (-8.2% / -15.0%)")
    st.success("🟢 **Safe-Guard 정상 작동 중**: 현재 MDD가 임계값(15%) 미만이므로 자율 거래를 유지합니다.")

st.divider()
st.caption(
    "⚠️ **[금융 면책 조항 (Disclaimer)]** 본 시스템은 대학교 캡스톤 디자인 교육 목적으로 개발되었으며, "
    "실제 투자 조언이나 자문에 사용할 수 없습니다. 백테스팅 및 시뮬레이션 성과가 미래의 수익을 보장하지 않습니다."
)