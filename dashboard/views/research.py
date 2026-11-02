"""에이전트 리서치: 질문 입력 → 추론 과정(Reasoning Trace) → 출처가 포함된 결과."""

from datetime import date, datetime, time, timezone

import pandas as pd
import streamlit as st

from core import api
from core.assets import DEFAULT_TICKERS, asset_name
from core.ui import error_card, page_header

NODE_LABELS = {
    "planner": ("계획", ":material/checklist:", "질문을 검색어로 분해"),
    "retrieve": ("검색", ":material/search:", "벡터DB에서 기준일 이전 문서 검색"),
    "grade": ("관련성 평가", ":material/rule:", "검색 결과가 충분한지 판단"),
    "rewrite": (
        "쿼리 재작성",
        ":material/edit_note:",
        "Self-Correction: 검색어 수정 후 재검색",
    ),
    "analyst": ("분석", ":material/psychology:", "근거를 인용해 의견 초안 작성"),
    "verifier": (
        "인용 검증",
        ":material/fact_check:",
        "주장마다 인용 문서가 뒷받침하는지 확인",
    ),
    "revise": ("초안 수정", ":material/edit:", "검증 실패 주장 재작성"),
    "finalize": ("최종 리포트", ":material/task_alt:", "검증된 주장만 남겨 결론"),
}
OPINION_STYLE = {
    "비중 확대": ("green", ":material/trending_up:"),
    "유지": ("gray", ":material/trending_flat:"),
    "비중 축소": ("red", ":material/trending_down:"),
}
SOURCE_LABELS = {"news": "뉴스", "dart": "DART 공시", "sec": "SEC 공시"}
EXAMPLES = [
    ("엔비디아 2024년 실적과 리스크 요인은?", ["NVDA"]),
    ("삼성전자 최근 실적과 리스크는?", ["005930"]),
    ("마이크로소프트 투자 의견은?", ["MSFT"]),
]

page_header(
    "에이전트 리서치",
    "LangGraph 에이전트가 뉴스·공시를 검색·검증해 출처가 포함된 투자 의견을 작성합니다",
)
history = st.session_state.setdefault("research_history", [])

st.caption("예시 질문")
ex_cols = st.columns(len(EXAMPLES))
for col, (q, t) in zip(ex_cols, EXAMPLES):
    if col.button(q, width="stretch", key=f"ex_{q}"):
        st.session_state["rq_question"], st.session_state["rq_tickers"] = q, t

with st.form("research_form", border=True):
    question = st.text_area(
        "투자 질문",
        key="rq_question",
        placeholder="예) 애플의 최근 실적 발표에서 주목할 리스크는?",
        height=80,
    )
    c1, c2 = st.columns([3, 1])
    tickers = c1.multiselect(
        "관련 자산 (비우면 질문에서 자동 인식)",
        DEFAULT_TICKERS,
        key="rq_tickers",
        format_func=asset_name,
        max_selections=12,
    )
    as_of = c2.date_input(
        "기준일",
        value=date(2024, 12, 31),
        help="이 시점 이전에 공개된 문서만 사용 (미래 정보 누수 방지)",
    )
    submitted = st.form_submit_button(
        "리서치 실행", type="primary", icon=":material/send:", width="stretch"
    )

if submitted:
    if len(question.strip()) < 2:
        st.warning("질문을 2자 이상 입력하세요.", icon=":material/edit:")
    else:
        as_of_iso = datetime.combine(as_of, time(0), tzinfo=timezone.utc).isoformat()
        with st.status("에이전트가 리서치 중입니다…", expanded=False) as status:
            st.write(
                "계획 → 검색 → 관련성 평가 → (필요 시 쿼리 재작성) → 분석 → 인용 검증 → 최종 리포트"
            )
            try:
                result = api.research(question.strip(), tickers, as_of_iso)
                history.insert(0, result)
                status.update(
                    label=f"완료 · {result.get('elapsed_ms', 0) / 1000:.2f}초",
                    state="complete",
                )
            except api.ApiError as error:
                status.update(label="리서치 실패", state="error")
                error_card(error)

if history:
    r = history[0]
    color, icon = OPINION_STYLE.get(r.get("opinion"), ("gray", ":material/help:"))
    with st.container(border=True):
        head, meta = st.columns([3, 2], vertical_alignment="center")
        with head:
            st.markdown(f"**Q. {r.get('question')}**")
            st.badge(r.get("opinion", "-"), icon=icon, color=color)
        with meta:
            st.progress(
                float(r.get("confidence", 0)),
                text=f"신뢰도 {r.get('confidence', 0):.2f}",
            )
            st.caption(
                f"자산 {', '.join(r.get('tickers', [])) or '-'} · "
                f"기준 {str(r.get('as_of', ''))[:10]} · "
                f"쿼리 재작성 {r.get('rewrites', 0)}회 · LLM `{r.get('llm_provider', '-')}`"
            )
        st.markdown(r.get("answer", ""))
        if r.get("insufficient_evidence"):
            st.warning(
                "기준일 이전 근거가 부족해 의견을 보수적으로 제시했습니다.",
                icon=":material/info:",
            )

    tab_trace, tab_claims, tab_risk = st.tabs(
        [
            ":material/route: 추론 과정",
            ":material/format_quote: 근거와 출처",
            ":material/crisis_alert: 리스크 태그",
        ]
    )
    with tab_trace:
        st.caption("에이전트의 단계별 '현재 생각' 로그 (Reasoning Trace)")
        for i, event in enumerate(r.get("trace", []), 1):
            name, node_icon, desc = NODE_LABELS.get(
                event["node"], (event["node"], ":material/circle:", "")
            )
            with st.expander(f"{i}. {name} — {event['message']}", icon=node_icon):
                st.caption(f"{desc} · {event.get('elapsed_ms', 0)} ms")
                if event.get("data"):
                    st.json(event["data"], expanded=False)

    with tab_claims:
        citations = {c["id"]: c for c in r.get("citations", [])}
        claims = r.get("claims", [])
        if claims:
            st.markdown("**검증된 주장**")
            for claim in claims:
                refs = " ".join(f"`[{n}]`" for n in claim.get("citations", []))
                st.markdown(f"- {claim['text']} {refs}")
        if citations:
            st.markdown("**원문 출처**")
            for cid, c in citations.items():
                with st.container(border=True):
                    a, b = st.columns([5, 1], vertical_alignment="center")
                    a.markdown(f"`[{cid}]` **{c['title']}**")
                    a.caption(
                        f"{SOURCE_LABELS.get(c['source_type'], c['source_type'])} · "
                        f"{asset_name(c['asset_id'])} · "
                        f"{str(c['published_at'])[:10]}"
                    )
                    b.link_button(
                        "원문",
                        c["source_url"],
                        icon=":material/open_in_new:",
                        width="stretch",
                    )
        else:
            st.info("인용된 출처가 없습니다.", icon=":material/info:")

    with tab_risk:
        tags = r.get("risk_tags", [])
        if tags:
            st.caption(
                "기준일에 유효한 위험 이벤트. 강화학습 관측 공간과 Safe-Guard에 반영됩니다."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "자산": asset_name(t["asset_id"]),
                            "유형": t["risk_type"],
                            "심각도": t["severity"],
                            "신뢰도": t["confidence"],
                            "유효 기간": (
                                f"{str(t['available_at'])[:10]} ~ "
                                f"{str(t['expires_at'])[:10]}"
                            ),
                            "출처": t["source_url"],
                        }
                        for t in tags
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "심각도": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.2f"
                    ),
                    "신뢰도": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.2f"
                    ),
                    "출처": st.column_config.LinkColumn(display_text="원문"),
                },
            )
        else:
            st.info(
                "기준일에 유효한 리스크 태그가 없습니다.",
                icon=":material/verified_user:",
            )

    if len(history) > 1:
        with st.expander(f"이전 질문 {len(history) - 1}건", icon=":material/history:"):
            for old in history[1:]:
                st.markdown(
                    f"- **{old.get('question')}** → {old.get('opinion')} "
                    f"(신뢰도 {old.get('confidence', 0):.2f})"
                )
else:
    st.info(
        "질문을 입력하고 **리서치 실행**을 누르세요. 예시 질문 버튼으로 바로 시작할 수 있습니다.",
        icon=":material/travel_explore:",
    )
