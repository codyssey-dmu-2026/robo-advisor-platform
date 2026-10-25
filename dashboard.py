import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8080")

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

@st.cache_data(ttl=15)
def get_backtest_data():
    resp = requests.get(f"{GATEWAY_URL}/backtest", timeout=5)
    resp.raise_for_status()
    return resp.json()

tabs = st.tabs([
    "1. 포트폴리오 현황",
    "2. 강화학습 성과",
    "3. SHAP 해석",
    "4. 에이전트 리서치",
    "5. ANOVA 검증 결과",
    "6. 리스크 모니터링"
])

# ==========================================
# [탭 1] 포트폴리오 현황 (POST /optimize)
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
# [탭 2] 강화학습 성과 (GET /backtest)
# ==========================================
with tabs[1]:
    st.subheader("📈 Walk-Forward 백테스트 및 12개 성과 지표 (`GET /backtest`)")
    try:
        bt_data = get_backtest_data()
        runs = bt_data.get("runs", {})

        rows = []
        for run_key, run_val in runs.items():
            metrics = run_val.get("metrics")
            if not metrics:
                continue
            row = {
                "Run ID": run_key,
                "전략(Strategy)": run_val.get("strategy"),
                "구간(Fold)": run_val.get("fold_id"),
                "관측일수": run_val.get("n_observations")
            }
            for m_name, m_obj in metrics.items():
                row[m_name] = m_obj.get("value") if isinstance(m_obj, dict) else m_obj
            rows.append(row)

        if rows:
            df_runs = pd.DataFrame(rows)
            st.markdown("#### 📋 Walk-Forward 윈도우별 12개 성과 지표 요약표")
            st.dataframe(df_runs.style.format(precision=4), use_container_width=True)

            metric_cols = [c for c in df_runs.columns if c not in ["Run ID", "전략(Strategy)", "구간(Fold)", "관측일수"]]
            sel_metric = st.selectbox("📊 시각화로 비교할 성과 지표 선택", options=metric_cols, index=metric_cols.index("sharpe") if "sharpe" in metric_cols else 0)

            fig_bt = px.bar(
                df_runs, x="구간(Fold)", y=sel_metric, color="전략(Strategy)",
                barmode="group", title=f"구간 및 전략별 {sel_metric} 비교"
            )
            st.plotly_chart(fig_bt, use_container_width=True)
        with st.expander("원본 백테스트 JSON 전체 보기"):
            st.json(bt_data)
    except Exception as e:
        st.warning(f"백테스트 데이터를 불러오지 못했습니다: {e}")

# ==========================================
# [탭 3] SHAP 해석 (POST /explain)
# ==========================================
with tabs[2]:
    st.subheader("🔍 SHAP 기반 의사결정 해석 (`POST /explain`)")
    st.caption("특정 시점(Decision ID)에서 각 피처 그룹(수익률·기술적지표·리스크)이 자산 비중 결정에 미친 기여도를 분석합니다.")

    col1, col2, col3 = st.columns(3)
    run_id = col1.selectbox("Run ID (`run_id`)", options=["toy_policy"])
    decision_id = col2.selectbox("의사결정 ID (`decision_id`)", options=["toy0", "toy1", "toy2"])
    target_asset = col3.selectbox("대상 자산 (`target_asset`)", options=[f"SYN0{i}" for i in range(10)])

    if st.button("SHAP 기여도 분석 조회", type="primary"):
        payload = {"run_id": run_id, "decision_id": decision_id, "target_asset": target_asset}
        try:
            res = requests.post(f"{GATEWAY_URL}/explain", json=payload, timeout=5)
            if res.status_code == 200:
                exp = res.json()
                st.success(f"`{run_id}` / `{decision_id}` / `{target_asset}` SHAP 해석 로드 성공!")

                m1, m2, m3 = st.columns(3)
                m1.metric("기본 비중 (Base Value)", f"{exp.get('base_value', 0):.4%}")
                m2.metric("예측 비중 (Predicted Weight)", f"{exp.get('predicted_weight', 0):.4%}")
                m3.metric("합산 오차 (Additivity Error)", f"{exp.get('additivity_error', 0):.2e}")

                # 피처 그룹별 SHAP 값 막대차트 시각화
                group_names = exp.get("group_names", ["returns", "indicators", "risk_inputs"])
                shap_vals = exp.get("shap_values") or exp.get("attributions") or exp.get("group_shap_values")
                if isinstance(shap_vals, dict):
                    df_shap = pd.DataFrame(list(shap_vals.items()), columns=["피처 그룹", "SHAP 기여도"])
                elif isinstance(shap_vals, list) and len(shap_vals) == len(group_names):
                    df_shap = pd.DataFrame({"피처 그룹": group_names, "SHAP 기여도": shap_vals})
                else:
                    df_shap = None

                if df_shap is not None:
                    fig_shap = px.bar(
                        df_shap, x="SHAP 기여도", y="피처 그룹", orientation="h",
                        color="SHAP 기여도", color_continuous_scale="RdBu_r",
                        title=f"{target_asset} 비중 결정에 대한 피처 그룹별 SHAP 기여도"
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    st.dataframe(df_shap, use_container_width=True)

                with st.expander("SHAP 응답 원본 JSON 보기", expanded=False):
                    st.json(exp)
            else:
                st.warning(f"응답 코드 {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"호출 실패: {e}")

# ==========================================
# [탭 4] 에이전트 리서치 (POST /research)
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
                            if r.get("trace"):
                                st.dataframe(pd.DataFrame(r["trace"]), use_container_width=True)
                        with col_b:
                            st.markdown("#### 🔗 검증된 원문 출처 (Source Citations)")
                            if r.get("citations"):
                                st.dataframe(pd.DataFrame(r["citations"]), use_container_width=True)

                        st.markdown("#### 🚨 시점 유효 리스크 태그 (`risk_tags`)")
                        if r.get("risk_tags"):
                            st.dataframe(pd.DataFrame(r["risk_tags"]), use_container_width=True)
                        else:
                            st.caption("해당 시점에 활성화된 리스크 태그가 없습니다.")
                    else:
                        st.error(f"오류 ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"통신 실패: {e}")

# ==========================================
# [탭 5] ANOVA 검증 결과 (GET /backtest)
# ==========================================
with tabs[4]:
    st.subheader("📐 ANOVA 통계적 유의성 검증 및 Paired Block Bootstrap")
    try:
        bt_data = get_backtest_data()
        anova_comp = bt_data.get("synthetic_baseline_comparison", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1️⃣ 일원분산분석 (ANOVA) 효과 검정")
            effects = anova_comp.get("effects", [])
            if effects:
                st.dataframe(pd.DataFrame(effects), use_container_width=True)
            diag = anova_comp.get("diagnostics", {})
            st.caption(f"등분산성(Levene p): `{diag.get('levene_p', 0):.4f}` | 정규성(Shapiro p): `{diag.get('shapiro_residual_p', 0):.4f}`")

        with col2:
            st.markdown("#### 2️⃣ 블록 부트스트랩 (Paired Circular Block Bootstrap)")
            boot = bt_data.get("bootstrap", [])
            if boot:
                st.dataframe(pd.DataFrame(boot), use_container_width=True)

        st.markdown("#### 3️⃣ 3대 미션 ANOVA 상태 (`mission_anova`)")
        mission = bt_data.get("mission_anova", {})
        if mission:
            st.dataframe(pd.DataFrame(mission).T, use_container_width=True)
    except Exception as e:
        st.warning(f"통계 데이터 조회 실패: {e}")

# ==========================================
# [탭 6] 리스크 모니터링 & Safe-Guard
# ==========================================
with tabs[5]:
    st.subheader("🚨 실시간 리스크 모니터링 및 Safe-Guard (MDD 15% 제한)")
    try:
        bt_data = get_backtest_data()
        runs = bt_data.get("runs", {})
        run_options = [k for k, v in runs.items() if "metrics" in v]
        sel_run = st.selectbox("모니터링할 백테스트 포트폴리오 선택", options=run_options, index=len(run_options)-1 if run_options else 0)

        m_dict = runs[sel_run]["metrics"]
        curr_mdd = m_dict["mdd"]["value"] * 100
        curr_var = m_dict["var_95"]["value"] * 100
        curr_cvar = m_dict["cvar_95"]["value"] * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("선택 구간 MDD (최대낙폭)", f"-{curr_mdd:.2f}%", "Safe-Guard 임계값: -15.00%")
        c2.metric("일간 VaR (95%)", f"-{curr_var:.2f}%")
        c3.metric("일간 CVaR (95%)", f"-{curr_cvar:.2f}%")

        ratio = min(int((curr_mdd / 15.0) * 100), 100)
        st.progress(ratio, text=f"MDD Safe-Guard 한도 소진율: {ratio}% (현재 -{curr_mdd:.2f}% / 한도 -15.00%)")

        if curr_mdd >= 15.0:
            st.error("🔴 **Safe-Guard 발동 (MDD ≥ 15%)**: 손실 한도를 초과하여 위험자산 비중을 축소하고 거래를 중단합니다!")
        else:
            st.success("🟢 **Safe-Guard 정상 운용 중 (MDD < 15%)**: 포트폴리오 낙폭이 안전 범위 내에 있습니다.")
    except Exception as e:
        st.info(f"백테스트 지표 연동 대기 중: {e}")

st.divider()
st.caption(
    "⚠️ **[금융 면책 조항 (Disclaimer)]** 본 시스템은 대학교 캡스톤 디자인 교육 목적으로 개발되었으며, "
    "실제 투자 조언이나 자문에 사용할 수 없습니다. 백테스팅 및 시뮬레이션 성과가 미래의 수익을 보장하지 않습니다."
)