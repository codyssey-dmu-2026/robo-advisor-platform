"""ANOVA 검증 결과: 과제 3대 검증 상태, 효과 검정, 블록 부트스트랩 신뢰구간."""

import pandas as pd
import streamlit as st

from core import api, charts
from core.ui import load, page_header, table_view

MISSION = {
    "reward_variants": ("검증 1", "보상 함수 3종 성과 비교", "One-way ANOVA"),
    "strategies": ("검증 2", "DRL vs MVO vs 동일가중", "One-way ANOVA"),
    "strategy_by_regime": (
        "검증 3",
        "시장 국면별(상승/횡보/하락) 전략 성과",
        "Two-way ANOVA",
    ),
}

bt = load(api.backtest)
page_header(
    "ANOVA 통계 검증",
    "전략 간 성과 차이가 우연인지 통계적으로 검증 (F·p-value·효과 크기 η²)",
    data_mode=bt.get("data_mode") if bt else None,
)

if bt:
    st.markdown("#### 과제 3대 검증")
    mission = bt.get("mission_anova", {})
    cols = st.columns(3)
    for col, (key, (tag, title, method)) in zip(cols, MISSION.items()):
        item = mission.get(key, {})
        done = item.get("status") == "complete"
        with col, st.container(border=True):
            st.markdown(f"**{tag} · {title}**")
            st.caption(method)
            if done:
                st.badge("완료", icon=":material/check_circle:", color="green")
            else:
                st.badge("결과 대기", icon=":material/pending:", color="gray")
                st.caption(item.get("reason", "실제 PPO 결과 연동 후 수행"))

    comp = bt.get("synthetic_baseline_comparison", {})
    st.markdown("#### 기준선 전략 비교 (현재 산출물)")
    if comp.get("interpretation"):
        st.caption(comp["interpretation"])
    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True):
        st.markdown("**효과 검정**")
        effects = pd.DataFrame(comp.get("effects", []))
        if not effects.empty:
            effects["유의성"] = effects["p_value"].map(
                lambda p: "✅ 유의 (p<0.05)" if p < 0.05 else "— 비유의"
            )
            st.dataframe(
                effects.rename(
                    columns={
                        "term": "요인",
                        "f": "F 통계량",
                        "p_value": "p-value",
                        "eta_squared": "효과 크기 η²",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "F 통계량": st.column_config.NumberColumn(format="%.3f"),
                    "p-value": st.column_config.NumberColumn(format="%.4f"),
                    "효과 크기 η²": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.caption(
                "η² 기준: 0.01 작음 · 0.06 중간 · 0.14 큼. "
                f"{comp.get('eta_squared_definition', '')}"
            )
        tukey = comp.get("tukey_status")
        if tukey:
            st.caption(f"사후 검정(Tukey HSD): {tukey}")
    with right, st.container(border=True):
        st.markdown("**가정 진단**")
        diag = comp.get("diagnostics", {})
        d1, d2 = st.columns(2)
        levene, shapiro = diag.get("levene_p"), diag.get("shapiro_residual_p")
        d1.metric(
            "등분산성 (Levene p)",
            "—" if levene is None else f"{levene:.3f}",
            help="0.05 이상이면 집단 간 분산이 같다는 가정을 기각하지 않음",
        )
        d2.metric(
            "정규성 (Shapiro p)",
            "—" if shapiro is None else f"{shapiro:.3f}",
            help="0.05 이상이면 잔차가 정규분포라는 가정을 기각하지 않음",
        )
        if comp.get("warning"):
            st.caption(f":material/warning: {comp['warning']}")

    boot = bt.get("bootstrap", [])
    if boot:
        with st.container(border=True):
            st.markdown("**블록 부트스트랩 · 월간 수익률 차이의 95% 신뢰구간**")
            st.caption(
                "구간이 0을 포함하면 차이가 통계적으로 뚜렷하지 않다는 뜻입니다."
            )
            labels = [f"비교 {i + 1}" for i in range(len(boot))]
            means = [b["mean_monthly_difference"] for b in boot]
            charts.show(
                charts.forest(
                    labels,
                    means,
                    [b["ci_95"][0] for b in boot],
                    [b["ci_95"][1] for b in boot],
                ),
                key="anova_forest",
            )
            table_view(
                pd.DataFrame(
                    [
                        {
                            "비교": label,
                            "평균 차이": b["mean_monthly_difference"],
                            "하한": b["ci_95"][0],
                            "상한": b["ci_95"][1],
                            "블록 길이": b["block_length"],
                            "반복": b["repetitions"],
                        }
                        for label, b in zip(labels, boot)
                    ]
                ),
                column_config={
                    c: st.column_config.NumberColumn(format="%.4f")
                    for c in ["평균 차이", "하한", "상한"]
                },
            )
