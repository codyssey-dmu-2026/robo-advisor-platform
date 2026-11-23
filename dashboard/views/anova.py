"""ANOVA 검증 결과: 과제 3대 검증의 수치·해석, 가정 진단, 사후 검정, 블록 부트스트랩."""

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
# 검정마다 같은 항 이름(C(strategy))이 다른 집단을 뜻하므로 검정별로 이름을 붙인다
TERM_LABELS = {
    "reward_variants": {"C(strategy)": "보상 함수"},
    "strategies": {"C(strategy)": "전략"},
    "strategy_by_regime": {
        "C(strategy, Sum)": "전략",
        "C(regime, Sum)": "시장 국면",
        "C(strategy, Sum):C(regime, Sum)": "전략 × 국면",
        "C(strategy)": "전략",
        "C(regime)": "시장 국면",
    },
}
# 비유의 해석에 쓸 비교 대상 이름
SUBJECTS = {
    "reward_variants": "보상 함수",
    "strategies": "전략(PPO·MVO·동일가중)",
    "strategy_by_regime": "전략",
}
SAMPLES = {
    "reward_variants": (
        "PPO 단순 수익률·샤프비율·MDD 페널티(λ={lam:.1f}, 기본값 고정) × 2022~2025년 "
        "48개월 월별 순수익률(seed 3개 평균)"
    ),
    "strategies": (
        "PPO(MDD 페널티 λ=1.0, seed 3개 평균) · MVO(매월 첫 결정일 리밸런싱, 최근 252개 "
        "수익률 공분산, 자산당 상한 40%) · 월간 동일가중 × 48개월"
    ),
    "strategy_by_regime": (
        "검증 2의 세 전략 × 시장 국면. 국면은 그달 시작 전 SPY 60개 관측의 누적 수익률로 "
        "판정(+5% 초과 상승, −5% 미만 하락, 나머지 횡보){counts}"
    ),
}
# 미완료 사유 (evaluation-xai 응답은 영어라 화면용으로 풀어 쓴다)
PENDING_REASON = {
    "strategies": "같은 실행 조건의 MVO 결과가 필요합니다 (evaluation-xai 대기).",
    "strategy_by_regime": "MVO 결과와 테스트 이전 국면 판정용 벤치마크가 필요합니다.",
}
COMPARISON_LABELS = {
    "PPO-mdd_l1.0_seed_mean_minus_EqualWeight_daily": "PPO(MDD λ=1.0) − 일간 동일가중",
    "PPO-mdd_l1.0_seed_mean_minus_EqualWeight_monthly": "PPO(MDD λ=1.0) − 월간 동일가중",
    "PPO-mdd_l1.0_seed_mean_minus_MVO_monthly": "PPO(MDD λ=1.0) − MVO",
}
REGIME_LABELS = {"up": "상승", "sideways": "횡보", "down": "하락"}


def eta_size(eta: float) -> str:
    if eta >= 0.14:
        return "큼"
    if eta >= 0.06:
        return "중간"
    if eta >= 0.01:
        return "작음"
    return "거의 없음"


def term_label(test: str, term: str) -> str:
    return TERM_LABELS.get(test, {}).get(term, term)


def significant(effect: dict) -> bool:
    return effect["p_value"] < 0.05


def effects_frame(test: str, effects: list[dict]) -> pd.DataFrame:
    rows = []
    for e in effects:
        row = {
            "요인": term_label(test, e["term"]),
            "F": e["f"],
            "p": e["p_value"],
            "η²": e["eta_squared"],
            "크기": eta_size(e["eta_squared"]),
        }
        if e.get("p_holm_family") is not None:
            row["Holm p"] = e["p_holm_family"]
        row["유의성"] = "✅ 유의" if significant(e) else "— 비유의"
        rows.append(row)
    return pd.DataFrame(rows)


def effects_table(test: str, effects: list[dict]) -> None:
    st.dataframe(
        effects_frame(test, effects),
        hide_index=True,
        width="stretch",
        column_config={
            "F": st.column_config.NumberColumn(format="%.3f", help="F 통계량"),
            "p": st.column_config.NumberColumn(format="%.4f", help="p-value"),
            "η²": st.column_config.NumberColumn(
                format="%.4f", help="효과 크기 (eta-squared)"
            ),
            "유의성": st.column_config.TextColumn(help="p < 0.05면 유의"),
            "Holm p": st.column_config.NumberColumn(
                format="%.4f",
                help="검정 3종의 효과 5개를 한 묶음으로 보고 여러 번 검정한 효과를 보정한 값",
            ),
        },
    )
    st.caption("η² 기준: 0.01 작음 · 0.06 중간 · 0.14 큼")


def tukey_caption(result: dict) -> str | None:
    status = result.get("tukey_status")
    pairs = result.get("tukey") or []
    if status == "not_required":
        return "모든 효과가 p ≥ 0.05라 사후 검정(Tukey HSD) 대상이 아닙니다."
    if pairs:
        hits = sum(1 for pair in pairs if pair.get("reject"))
        return (
            f"p < 0.05인 효과가 있어 사후 검정(Tukey HSD)으로 {len(pairs)}쌍을 비교했고, "
            f"유의한 쌍은 {hits}개입니다."
        )
    return f"사후 검정(Tukey HSD): {status}" if status else None


def diagnostics(result: dict) -> None:
    st.markdown("**가정 진단**")
    diag = result.get("diagnostics", {})
    d1, d2 = st.columns(2)
    levene, shapiro = diag.get("levene_p"), diag.get("shapiro_residual_p")
    d1.metric(
        "등분산성 (Levene p)",
        "—" if levene is None else ("<0.001" if levene < 0.001 else f"{levene:.3f}"),
        help="0.05 이상이면 집단 간 분산이 같다는 가정을 기각하지 않음",
    )
    d2.metric(
        "정규성 (Shapiro p)",
        "—" if shapiro is None else f"{shapiro:.3f}",
        help="0.05 이상이면 잔차가 정규분포라는 가정을 기각하지 않음",
    )
    if levene is not None and levene < 0.05:
        st.caption(
            ":material/warning: 등분산 가정이 기각됩니다(p < 0.05). "
            "집단마다 수익률 변동 폭이 달라 F 검정 결과를 보수적으로 읽습니다."
        )
    st.caption(
        ":material/warning: 같은 달을 반복 사용하고 월 수익률에 자기상관이 있어 "
        "ANOVA만으로는 보정되지 않습니다. 아래 블록 부트스트랩을 보조 근거로 봅니다."
    )


def interpretation(test: str, result: dict) -> str:
    """과제 규칙: 유의하면 효과를, 비유의면 효과 크기 해석과 원인을 함께 적는다."""
    effects = result["effects"]
    hits = [e for e in effects if significant(e)]
    if not hits:
        e = effects[0]
        p, eta = e["p_value"], e["eta_squared"]
        return (
            f"p = {p:.3f}로 {SUBJECTS[test]} 간 월평균 수익률 차이가 우연 범위를 벗어나지 "
            f"않습니다. 효과 크기 η² = {eta:.4f}는 '{eta_size(eta)}' 수준으로, 수익률 변동 "
            f"가운데 집단 차이로 설명되는 부분이 {eta:.1%}에 불과합니다. 모두 같은 12개 "
            "자산·같은 시장 구간에서 운용하고 월 수익률 변동의 대부분이 시장 전체 움직임이라, "
            f"{SUBJECTS[test]} 차이가 묻히는 것으로 해석합니다."
        )
    parts = []
    for e in hits:
        holm = e.get("p_holm_family")
        holm_text = f", Holm 보정 p = {holm:.4f}" if holm is not None else ""
        parts.append(
            f"{term_label(test, e['term'])} 효과가 유의합니다(p = {e['p_value']:.4f}"
            f"{holm_text}, η² = {e['eta_squared']:.3f} '{eta_size(e['eta_squared'])}')"
        )
    rest = [
        f"{term_label(test, e['term'])}(p = {e['p_value']:.3f})"
        for e in effects
        if not significant(e)
    ]
    text = ". ".join(parts) + "."
    if test == "strategy_by_regime" and any(
        e["term"].startswith("C(regime") for e in hits
    ):
        text += " 상승장·횡보장·하락장 여부가 월 수익률을 크게 좌우한다는 뜻입니다."
    if rest:
        text += f" 반면 {', '.join(rest)}은 비유의라, 국면과 상관없이 전략 간 차이는 뚜렷하지 않습니다."
    tukey = tukey_caption(result)
    if tukey and result.get("tukey"):
        text += " " + tukey
    return text


def mismatch_caption(result: dict) -> str | None:
    recon = result.get("input_reconciliation") or {}
    runs = recon.get("mismatched_runs") or []
    if recon.get("status") == "includes_reference_mismatch" and runs:
        return (
            f":material/info: 이 검정에 쓴 {recon.get('run_count', '?')}개 실행 중 "
            f"{len(runs)}개({', '.join(runs)})는 rl-engine 공개 결과와 엄격한 허용오차"
            "(1e-8)를 넘게 다릅니다. 차이는 float32 계산 수준이며 evaluation-xai가 원인을 기록해 두었습니다."
        )
    return None


bt = load(api.backtest)
mode = bt.get("data_mode") if bt else None
if bt and mode == "real" and str(bt.get("status", "")).startswith("provisional"):
    mode = "imported_unverified"  # 실제 결과지만 최종 검증 전이라 '잠정'으로 표시
page_header(
    "ANOVA 통계 검증",
    "전략 간 성과 차이가 우연인지 통계적으로 검증 (F·p-value·효과 크기 η²)",
    data_mode=mode,
)

if bt:
    mission = bt.get("mission_anova", {})
    st.markdown("#### 과제 3대 검증")
    cols = st.columns(3)
    for col, (key, (tag, title, method)) in zip(cols, MISSION.items()):
        item = mission.get(key, {})
        effects = item.get("effects") or []
        done = item.get("status") == "complete" and effects
        with col, st.container(border=True):
            st.markdown(f"**{tag} · {title}**")
            st.caption(method)
            if done:
                with st.container(horizontal=True):
                    st.badge("완료", icon=":material/check_circle:", color="green")
                    if any(significant(e) for e in effects):
                        st.badge(
                            "유의한 효과 있음", icon=":material/star:", color="blue"
                        )
                for e in effects:
                    mark = "✅ " if significant(e) else ""
                    st.caption(
                        f"{mark}{term_label(key, e['term'])}: F = {e['f']:.3f} · "
                        f"p = {e['p_value']:.3f} · η² = {e['eta_squared']:.4f}"
                    )
            else:
                st.badge("결과 대기", icon=":material/pending:", color="gray")
                st.caption(PENDING_REASON.get(key, item.get("reason", "결과 대기")))
    if mission.get("holm_status") == "complete":
        st.caption(
            f"Holm 보정: 검정 3종의 효과 {mission.get('holm_family_size', 5)}개를 한 묶음으로 "
            "보고 보정했습니다(여러 번 검정해서 우연히 유의해 보이는 것을 막기 위함)."
        )

    for key, (tag, title, method) in MISSION.items():
        result = mission.get(key, {})
        if result.get("status") != "complete" or not result.get("effects"):
            continue
        st.markdown(f"#### {tag} · {title}")
        counts = mission.get("regime_month_counts") or {}
        count_text = (
            " · "
            + ", ".join(
                f"{REGIME_LABELS.get(k, k)} {v}개월"
                for k, v in counts.items()
                if k in REGIME_LABELS
            )
            if counts
            else ""
        )
        st.caption(
            "표본: "
            + SAMPLES[key].format(
                lam=result.get("representative_mdd_lambda", 1.0), counts=count_text
            )
        )
        left, right = st.columns([3, 2], gap="medium")
        with left, st.container(border=True):
            st.markdown("**효과 검정**")
            effects_table(key, result["effects"])
            tukey = tukey_caption(result)
            if tukey:
                st.caption(tukey)
            if result.get("tukey"):
                with st.expander(f"사후 검정 {len(result['tukey'])}쌍 보기"):
                    st.dataframe(
                        pd.DataFrame(result["tukey"]).rename(
                            columns={
                                "group1": "집단 1",
                                "group2": "집단 2",
                                "mean_difference": "평균 차이",
                                "p_adjusted": "보정 p",
                                "lower": "하한",
                                "upper": "상한",
                                "reject": "유의",
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )
        with right, st.container(border=True):
            diagnostics(result)
        st.info(interpretation(key, result), icon=":material/lightbulb:")
        note = mismatch_caption(result)
        if note:
            st.caption(note)

    # 합성 데모 산출물에만 있는 기준선 비교 (실데이터 산출물에는 없음)
    comp = bt.get("synthetic_baseline_comparison", {})
    if comp.get("effects"):
        st.markdown("#### 기준선 전략 비교 (합성 데모)")
        if comp.get("interpretation"):
            st.caption(comp["interpretation"])
        left, right = st.columns([3, 2], gap="medium")
        with left, st.container(border=True):
            st.markdown("**효과 검정**")
            effects_table("strategies", comp["effects"])
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
            names = [
                COMPARISON_LABELS.get(b.get("comparison"), f"비교 {i + 1}")
                for i, b in enumerate(boot)
            ]
            labels = [
                f"{n} · 블록 {b['block_length']}개월" for n, b in zip(names, boot)
            ]
            charts.show(
                charts.forest(
                    labels,
                    [b["mean_monthly_difference"] for b in boot],
                    [b["ci_95"][0] for b in boot],
                    [b["ci_95"][1] for b in boot],
                    height=max(260, 60 * len(boot)),
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
            for name in dict.fromkeys(names):
                group = [b for n, b in zip(names, boot) if n == name]
                mean = group[0]["mean_monthly_difference"]
                if all(b["ci_95"][1] < 0 for b in group):
                    verdict = "모든 블록 길이에서 신뢰구간이 0보다 작아, PPO가 뚜렷하게 낮습니다"
                elif all(b["ci_95"][0] > 0 for b in group):
                    verdict = "모든 블록 길이에서 신뢰구간이 0보다 커, PPO가 뚜렷하게 높습니다"
                else:
                    verdict = "신뢰구간이 0을 포함해, 차이가 뚜렷하지 않습니다"
                st.caption(
                    f":material/info: {name}: 월평균 {mean * 100:+.2f}%p, {verdict}."
                )

    if str(bt.get("status", "")).startswith("provisional"):
        st.caption(
            ":material/pending_actions: evaluation-xai가 이 결과를 '잠정'으로 표시했습니다. "
            "통계 계산은 끝났지만 가격·환율·무위험수익률·시장별 시각 검증 등 최종 평가 조건이 남아 있습니다."
        )
    limitations = bt.get("limitations") or []
    if limitations:
        with st.expander(
            f"evaluation-xai가 밝힌 분석 한계 {len(limitations)}건 (원문)",
            icon=":material/rule:",
        ):
            st.markdown("\n".join(f"- {item}" for item in limitations))
