"""강화학습 성과: 과제 성능 기준 점검, 보상 함수·벤치마크 비교, 12개 지표, 학습 곡선."""

import pandas as pd
import streamlit as st

from core import api, charts
from core.ui import (
    METRIC_LABELS,
    empty_state,
    load,
    page_header,
    perf_entries,
    perf_label,
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
        empty_state(
            "학습 곡선은 rl-engine 학습 기록을 받는 중입니다.",
            "PPO 학습 중 Episode Reward 기록(TensorBoard 로그)이 공유되면 "
            "보상 함수별(단순 수익률·샤프·MDD 페널티) 수렴 곡선이 이곳에 표시됩니다.",
            icon=":material/model_training:",
        )

    st.caption(f":material/database: 출처: {perf['source']}")
