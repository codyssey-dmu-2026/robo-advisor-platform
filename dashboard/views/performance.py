"""강화학습 성과: 과제 성능 기준 점검, 보상 함수·벤치마크 비교, 12개 지표, 학습 곡선."""

import pandas as pd
import streamlit as st

from core import api, charts
from core.ui import (
    METRIC_LABELS,
    load,
    page_header,
    perf_entries,
    perf_label,
    table_view,
    window_label,
)

# 과제 성능 기준 (assignment: 벤치마크 대비 샤프 +0.2 이상, 누적 수익률 +10%p 이상)
SHARPE_TARGET = 0.2
RETURN_TARGET = 0.10

perf = load(api.performance)
page_header(
    "강화학습 성과",
    "Walk-Forward 검증(구간마다 재학습 → 다음 1년 테스트)으로 본 전략별 성과",
    data_mode=perf.get("data_mode") if perf else None,
)


def fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "—"
    return f"{value:.2%}" if kind == "pct" else f"{value:.3f}"


# 학습 곡선: 과제 필수 3종을 윗줄, MDD λ 탐색 2개를 아랫줄에 둔다
VARIANTS = {
    "simple": "단순 수익률 (r_t)",
    "sharpe": "샤프비율 (r_t / σ_t)",
    "mdd_l1.0": "MDD 페널티 λ=1.0",
    "mdd_l0.5": "MDD 페널티 λ=0.5",
    "mdd_l5.0": "MDD 페널티 λ=5.0",
}
STAGES = {"test": "최종 모델 학습", "val": "선택용 학습 (검증 구간)"}


def learning_curve_tab(perf: dict) -> None:
    curves = load(api.learning_curves, spinner="학습 곡선 불러오는 중…")
    if not curves:
        return
    runs = curves["runs"]
    current = next((r for r in runs if r["run"] == curves["current_model"]), None)
    c1, c2 = st.columns(2)
    stage = c1.segmented_control(
        "학습 단계",
        list(STAGES),
        default="test",
        format_func=STAGES.get,
        required=True,
        key="lc_stage",
        help="최종 모델: 학습 구간 전체로 학습해 다음 해에 테스트한 모델. "
        "선택용: 학습 구간 마지막 6개월을 떼어 보상 함수·설정을 고를 때 쓴 모델",
    )
    windows = sorted({r["window"] for r in runs})
    window = c2.segmented_control(
        "테스트 연도 (Walk-Forward 구간)",
        windows,
        default=current["window"] if current else windows[-1],
        format_func=lambda w: window_label(perf, w),
        required=True,
        key="lc_window",
    )
    st.caption(
        f":material/info: {curves['note']} "
        "가로축은 학습 스텝(최소 기준 100,000), 세로축은 최근 에피소드의 평균 보상입니다."
    )
    if current:
        st.caption(
            f"현재 서빙 모델 `{current['run']}`: 평균 보상 "
            f"{current['start_reward']:.3f} → {current['end_reward']:.3f} "
            f"({current['total_steps']:,} 스텝)"
        )

    chosen = [r for r in runs if r["stage"] == stage and r["window"] == window]
    seeds = sorted({r["seed"] for r in chosen})
    colors = charts.color_map([f"seed {s}" for s in seeds])
    keys = list(VARIANTS)
    for row in (keys[:3], keys[3:]):
        if row == keys[3:]:
            st.caption("MDD 페널티 강도(λ) 탐색")
        for col, variant in zip(st.columns(3), row):
            lines = {
                f"seed {r['seed']}": (r["steps"], r["ep_rew_mean"])
                for r in sorted(chosen, key=lambda r: r["seed"])
                if r["variant"] == variant
            }
            with col, st.container(border=True):
                st.markdown(f"**{VARIANTS[variant]}**")
                if lines:
                    charts.show(
                        charts.line(lines, colors=colors, percent_y=False, height=240),
                        key=f"lc_{stage}_{window}_{variant}",
                    )
                else:
                    st.caption("기록 없음")

    table_view(
        pd.DataFrame(
            [
                {
                    "보상 함수": VARIANTS.get(r["variant"], r["variant"]),
                    "seed": r["seed"],
                    "시작 평균 보상": r["start_reward"],
                    "마지막 평균 보상": r["end_reward"],
                    "변화": r["end_reward"] - r["start_reward"],
                    "마지막 평균 에피소드 길이": r["end_episode_length"],
                    "학습 스텝": r["total_steps"],
                }
                for r in sorted(
                    chosen, key=lambda r: (keys.index(r["variant"]), r["seed"])
                )
            ]
        ),
        label="표로 보기 (처음·마지막 5개 기록 평균)",
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ["시작 평균 보상", "마지막 평균 보상", "변화"]
        }
        | {
            "마지막 평균 에피소드 길이": st.column_config.NumberColumn(
                format="%.0f일",
                help="학습 중 낙폭이 15%를 넘으면 에피소드가 조기 종료(Safe-Guard)됩니다",
            )
        },
    )


if perf:
    entries = perf_entries(perf)
    model = entries["model"]["total"]
    labels = [perf_label(e) for e in perf["strategies"]]
    colors = charts.color_map(labels)
    windows = [window_label(perf, w["window"]) for w in perf["windows"]]

    # ---------------- 과제 성능 기준 ----------------
    with st.container(border=True):
        st.markdown("**과제 성능 기준 점검 · 4년 전체**")
        rows = []
        for kind, name in (("benchmark", "벤치마크(SPY)"), ("baseline", "동일가중")):
            other = entries[kind]["total"]
            for key, target, label, unit in (
                ("sharpe", SHARPE_TARGET, "샤프 비율", ""),
                ("cumulative_return", RETURN_TARGET, "누적 수익률", "%p"),
            ):
                diff = model[key] - other[key]
                shown = f"{diff * 100:+.1f}%p" if unit else f"{diff:+.2f}"
                goal = f"+{target * 100:.0f}%p" if unit else f"+{target}"
                rows.append(
                    {
                        "비교 대상": name,
                        "지표": label,
                        "현재 모델": fmt(model[key], "pct" if unit else "num"),
                        "비교 대상 값": fmt(other[key], "pct" if unit else "num"),
                        "차이": shown,
                        "목표": f"{goal} 이상",
                        "판정": "✅ 달성" if diff >= target else "❌ 미달",
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"현재 모델 `{perf['model_name']}`: 검증 구간 성적으로 보상 함수를 고른 모델 "
            "(시험 구간 성적을 보고 고르면 미래 정보 누수가 되므로). "
            "API 응답 시간 기준(5초 이내)은 테스트 코드로 확인합니다."
        )

    tab_cmp, tab_table, tab_curve = st.tabs(
        [
            ":material/bar_chart: 전략 비교",
            ":material/table: 12개 지표 전체",
            ":material/show_chart: 학습 곡선",
        ]
    )

    with tab_cmp:
        metric = st.pills(
            "지표",
            list(METRIC_LABELS),
            default="sharpe",
            format_func=lambda k: METRIC_LABELS[k][0],
            required=True,
        )
        label, kind = METRIC_LABELS[metric]
        with st.container(border=True):
            st.markdown(f"**연도별 {label}**")
            groups = {}
            for entry in perf["strategies"]:
                by_window = {
                    w["window"]: w["metrics"].get(metric) for w in entry["windows"]
                }
                groups[perf_label(entry)] = [
                    by_window.get(w["window"]) for w in perf["windows"]
                ]
            charts.show(
                charts.grouped_bar(
                    windows, groups, colors=colors, percent=kind == "pct"
                ),
                key="perf_bar",
            )
        with st.container(border=True):
            st.markdown(f"**4년 전체 {label}**")
            totals = sorted(
                (
                    (perf_label(e), e["total"].get(metric))
                    for e in perf["strategies"]
                    if e["total"].get(metric) is not None
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
            charts.show(
                charts.hbar(
                    [n for n, _ in totals],
                    [v for _, v in totals],
                    percent=kind == "pct",
                ),
                key="perf_total",
            )
        st.caption(
            "다른 보상 함수는 시드 3개 평균, 현재 모델은 고정된 시드 1개 결과입니다."
        )

    with tab_table:
        config = {
            METRIC_LABELS[k][0]: st.column_config.NumberColumn(
                format="percent" if kind == "pct" else "%.3f"
            )
            for k, (_, kind) in METRIC_LABELS.items()
        }

        def metric_row(name: str, period: str, metrics: dict) -> dict:
            row = {"전략": name, "기간": period}
            row.update({METRIC_LABELS[k][0]: metrics.get(k) for k in METRIC_LABELS})
            return row

        st.markdown("**4년 전체**")
        st.dataframe(
            pd.DataFrame(
                [
                    metric_row(perf_label(e), "4년", e["total"])
                    for e in perf["strategies"]
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config=config,
        )
        with st.expander("연도별 지표", icon=":material/calendar_month:"):
            st.dataframe(
                pd.DataFrame(
                    [
                        metric_row(
                            perf_label(e),
                            window_label(perf, w["window"]),
                            w["metrics"],
                        )
                        for e in perf["strategies"]
                        for w in e["windows"]
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config=config,
            )
        st.caption(
            "알파·베타·정보비율은 SPY 대비 값입니다. SPY 자신은 알파 0·베타 1이고 "
            "정보비율은 정의되지 않아 비워 둡니다."
        )

    with tab_curve:
        learning_curve_tab(perf)

    st.caption(f":material/database: 출처: {perf['source']}")
