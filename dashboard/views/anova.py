"""ANOVA 검증 결과: 과제 3대 검증 상태와 수치, 가정 진단, 블록 부트스트랩 신뢰구간."""

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
# 미완료 사유 (evaluation-xai 응답은 영어라 화면용으로 풀어 쓴다)
PENDING_REASON = {
    "strategies": "같은 실행 조건의 MVO 결과가 필요합니다 (evaluation-xai 대기).",
    "strategy_by_regime": "MVO 결과와 테스트 이전 국면 판정용 벤치마크가 필요합니다.",
}
TUKEY_STATUS = {
    "not_required": "p ≥ 0.05라 사후 검정(Tukey HSD) 대상이 아닙니다.",
    "complete": "p < 0.05라 사후 검정(Tukey HSD)을 수행했습니다.",
}
TERM_LABELS = {"C(strategy)": "보상 함수", "C(regime)": "시장 국면"}
COMPARISON_LABELS = {
    "PPO-mdd_l1.0_seed_mean_minus_EqualWeight_daily": "PPO(MDD λ=1.0) − 동일가중",
}


def eta_size(eta: float) -> str:
    if eta >= 0.14:
        return "큼"
    if eta >= 0.06:
        return "중간"
    if eta >= 0.01:
        return "작음"
    return "거의 없음"


def effects_frame(effects: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(effects)
    if frame.empty:
        return frame
    frame["term"] = frame["term"].map(lambda t: TERM_LABELS.get(t, t))
    frame["유의성"] = frame["p_value"].map(
        lambda p: "✅ 유의 (p<0.05)" if p < 0.05 else "— 비유의"
    )
    frame["효과 크기"] = frame["eta_squared"].map(eta_size)
    return frame.rename(
        columns={
            "term": "요인",
            "f": "F 통계량",
            "p_value": "p-value",
            "eta_squared": "효과 크기 η²",
        }
    )


def effects_table(effects: list[dict]) -> None:
    st.dataframe(
        effects_frame(effects),
        hide_index=True,
        width="stretch",
        column_config={
            "F 통계량": st.column_config.NumberColumn(format="%.3f"),
            "p-value": st.column_config.NumberColumn(format="%.4f"),
            "효과 크기 η²": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption("η² 기준: 0.01 작음 · 0.06 중간 · 0.14 큼")


def diagnostics(result: dict) -> None:
    st.markdown("**가정 진단**")
    diag = result.get("diagnostics", {})
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


def interpretation(effect: dict) -> str:
    """비유의 결과는 과제 규칙대로 효과 크기 해석과 원인을 함께 적는다."""
    p, eta = effect["p_value"], effect["eta_squared"]
    if p < 0.05:
        return (
            f"p = {p:.4f}로 집단 간 평균 차이가 통계적으로 유의합니다. "
            f"효과 크기 η² = {eta:.4f} ({eta_size(eta)})."
        )
    return (
        f"p = {p:.3f}로 집단 간 월평균 수익률 차이가 우연 범위를 벗어나지 않습니다. "
        f"효과 크기 η² = {eta:.4f}는 '{eta_size(eta)}' 수준으로, 수익률 변동 가운데 "
        f"집단 차이로 설명되는 부분이 {eta:.1%}에 불과합니다. "
        "세 모델이 같은 12개 자산·같은 시장 구간에서 비슷한 비중을 내는 데다 "
        "월 수익률 변동(시장 요인)이 커서, 보상 함수 차이가 묻히는 것으로 해석합니다."
    )


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
        effects = item.get("effects") or []
        done = item.get("status") == "complete" and effects
        with col, st.container(border=True):
            st.markdown(f"**{tag} · {title}**")
            st.caption(method)
            if done:
                st.badge("완료", icon=":material/check_circle:", color="green")
                e = effects[0]
                st.caption(
                    f"F = {e['f']:.3f} · p = {e['p_value']:.3f} · "
                    f"η² = {e['eta_squared']:.4f}"
                )
            else:
                st.badge("결과 대기", icon=":material/pending:", color="gray")
                st.caption(PENDING_REASON.get(key, item.get("reason", "결과 대기")))

    for key, (tag, title, method) in MISSION.items():
        result = mission.get(key, {})
        if result.get("status") != "complete" or not result.get("effects"):
            continue
        st.markdown(f"#### {tag} · {title}")
        if key == "reward_variants":
            lam = result.get("representative_mdd_lambda", 1.0)
            st.caption(
                "표본: rl-engine 4개 테스트 연도(2022~2025)의 월별 순수익률, seed 3개 평균. "
                f"보상 함수는 단순 수익률·샤프비율·MDD 페널티(λ={lam:g}, 기본값 고정)."
            )
        left, right = st.columns([3, 2], gap="medium")
        with left, st.container(border=True):
            st.markdown("**효과 검정**")
            effects_table(result["effects"])
            tukey = result.get("tukey_status")
            if tukey:
                st.caption(TUKEY_STATUS.get(tukey, f"사후 검정(Tukey HSD): {tukey}"))
        with right, st.container(border=True):
            diagnostics(result)
            st.caption(
                ":material/warning: 같은 달을 반복 사용하고 월 수익률에 자기상관이 있어 "
                "ANOVA만으로는 보정되지 않습니다. 아래 블록 부트스트랩을 보조 근거로 봅니다."
            )
        st.info(interpretation(result["effects"][0]), icon=":material/lightbulb:")

    # 합성 데모 산출물에만 있는 기준선 비교 (실데이터 산출물에는 없음)
    comp = bt.get("synthetic_baseline_comparison", {})
    if comp.get("effects"):
        st.markdown("#### 기준선 전략 비교 (합성 데모)")
        if comp.get("interpretation"):
            st.caption(comp["interpretation"])
        left, right = st.columns([3, 2], gap="medium")
        with left, st.container(border=True):
            st.markdown("**효과 검정**")
            effects_table(comp["effects"])
            if comp.get("tukey_status"):
                st.caption(f"사후 검정(Tukey HSD): {comp['tukey_status']}")
        with right, st.container(border=True):
            diagnostics(comp)

    boot = bt.get("bootstrap", [])
    if boot:
        with st.container(border=True):
            st.markdown("**블록 부트스트랩 · 월간 수익률 차이의 95% 신뢰구간**")
            st.caption(
                "같은 연도 안에서 연속한 달을 묶어 다시 뽑아 자기상관을 보존합니다. "
                "구간이 0을 포함하면 차이가 통계적으로 뚜렷하지 않다는 뜻입니다."
            )
            labels = [
                f"{COMPARISON_LABELS.get(b.get('comparison'), f'비교 {i + 1}')}"
                f" · 블록 {b['block_length']}개월"
                for i, b in enumerate(boot)
            ]
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
            if all(b["ci_95"][1] < 0 for b in boot):
                st.caption(
                    ":material/info: 모든 블록 길이에서 신뢰구간 전체가 0보다 작습니다. "
                    "이 비교에서는 첫 번째 전략의 월평균 수익률이 더 낮습니다."
                )

    limitations = bt.get("limitations") or []
    if limitations:
        with st.expander(
            f"evaluation-xai가 밝힌 분석 한계 {len(limitations)}건 (원문)",
            icon=":material/rule:",
        ):
            st.markdown("\n".join(f"- {item}" for item in limitations))
