"""통합 자율 로보어드바이저 대시보드 진입점.

    streamlit run dashboard/app.py

대시보드는 모델 파일을 직접 로드하지 않고 API Gateway와 HTTP로만 통신한다 (MSA 제약).
"""

from pathlib import Path

import streamlit as st

from core import api
from core.ui import SERVICE_LABELS, footer

st.set_page_config(page_title="Robo-Advisor | 통합 자율 로보어드바이저", page_icon=":material/insights:", layout="wide")

PAGES = {
    "": [st.Page("views/home.py", title="개요", icon=":material/space_dashboard:", default=True)],
    "투자": [
        st.Page("views/portfolio.py", title="포트폴리오 현황", icon=":material/donut_large:", url_path="portfolio"),
        st.Page("views/risk.py", title="리스크 모니터링", icon=":material/shield:", url_path="risk"),
    ],
    "분석": [
        st.Page("views/performance.py", title="강화학습 성과", icon=":material/monitoring:", url_path="performance"),
        st.Page("views/shap.py", title="SHAP 해석", icon=":material/account_tree:", url_path="shap"),
        st.Page("views/anova.py", title="ANOVA 검증 결과", icon=":material/functions:", url_path="anova"),
    ],
    "리서치": [
        st.Page("views/research.py", title="에이전트 리서치", icon=":material/travel_explore:", url_path="research"),
    ],
}


def sidebar_status() -> None:
    st.markdown("##### :material/hub: 시스템 상태")
    try:
        services = api.health().get("services", {})
    except api.ApiError as error:
        st.error(f"API Gateway 오프라인\n\n{error.hint or ''}", icon=":material/cloud_off:")
        return
    for key, (name, desc) in SERVICE_LABELS.items():
        info = services.get(key, {})
        ok = info.get("status") == "ok"
        icon, label = (":material/check_circle:", "정상") if ok else (":material/error:", info.get("status", "unknown"))
        st.markdown(f"{icon} **{name}** · {label}  \n<span style='opacity:.65;font-size:.8rem'>{desc}</span>",
                    unsafe_allow_html=True)
    rl = services.get("rl_engine", {})
    if rl.get("status") == "ok" and rl.get("is_mock", True):
        st.caption(":material/info: 강화학습 모델 연동 전: 비중은 시뮬레이션 값입니다.")
    if st.button("상태 새로고침", icon=":material/refresh:", width="stretch"):
        api.health.clear()
        st.rerun()


# 로고는 사이드바 맨 위(메뉴 위)에 고정. 사이드바를 접으면 아이콘만 표시
STATIC = Path(__file__).resolve().parent / "static"
st.logo(str(STATIC / "logo.svg"), size="large", icon_image=str(STATIC / "logo_icon.svg"))

page = st.navigation(PAGES, position="sidebar", expanded=True)

with st.sidebar:
    st.caption("에이전틱 RAG × 강화학습 통합 자율 로보어드바이저")
    st.divider()
    sidebar_status()
    st.divider()
    st.caption(":material/palette: 테마 변경: 우측 상단 ⋮ 메뉴 맨 위의 System · Light · Dark")

page.run()
footer()
