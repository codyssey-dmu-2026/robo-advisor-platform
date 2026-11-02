"""강화학습 성과: 학습 곡선, Walk-Forward 백테스트 12개 지표, 전략 비교."""

import pandas as pd
import streamlit as st

from core import api, charts
from core.ui import (
    METRIC_LABELS,
    empty_state,
    evaluated_runs,
    load,
    metric_value,
    page_header,
    strategy_label,
)

bt = load(api.backtest)
page_header(
    "강화학습 성과",
    "Walk-Forward 검증(학습 4년 → 테스트 1년)으로 본 전략별 성과",
    data_mode=bt.get("data_mode") if bt else None,
)

if bt:
    runs = evaluated_runs(bt)
    strategies = sorted({r["strategy"] for _, r in runs})
    folds = sorted({r["fold_id"] for _, r in runs})
    colors = {strategy_label(s): c for s, c in charts.color_map(strategies).items()}

    tab_cmp, tab_table, tab_curve = st.tabs(
        [
            ":material/bar_chart: 전략 비교",
            ":material/table: 12개 지표 전체",
            ":material/show_chart: 학습 곡선",
        ]
    )

    with tab_cmp:
        keys = list(METRIC_LABELS)
        metric = st.pills(
            "지표",
            keys,
            default="sharpe",
            format_func=lambda k: METRIC_LABELS[k][0],
            required=True,
        )
        label, kind = METRIC_LABELS[metric]
        groups = {}
        for s in strategies:
            by_fold = {
                r["fold_id"]: metric_value(r, metric)
                for _, r in runs
                if r["strategy"] == s
            }
            groups[strategy_label(s)] = [by_fold.get(f) for f in folds]
        with st.container(border=True):
            st.markdown(f"**구간별 {label}**")
            charts.show(
                charts.grouped_bar(folds, groups, colors=colors, percent=kind == "pct"),
                key="perf_bar",
            )
        averages = {
            name: [v for v in vals if v is not None] for name, vals in groups.items()
        }
        cols = st.columns(len(averages) or 1)
        for col, (name, vals) in zip(cols, averages.items()):
            avg = sum(vals) / len(vals) if vals else None
            col.metric(
                f"{name} · 평균 {label}",
                (
                    "—"
                    if avg is None
                    else (f"{avg:.2%}" if kind == "pct" else f"{avg:.3f}")
                ),
                border=True,
            )

    with tab_table:
        rows = []
        for run_id, r in runs:
            row = {
                "전략": strategy_label(r["strategy"]),
                "구간": r["fold_id"],
                "기간": f"{r.get('period_start', '')} ~ {r.get('period_end', '')}",
                "관측일": r.get("n_observations"),
            }
            row.update({METRIC_LABELS[k][0]: metric_value(r, k) for k in METRIC_LABELS})
            rows.append(row)
        config = {
            METRIC_LABELS[k][0]: st.column_config.NumberColumn(
                format="percent" if kind == "pct" else "%.3f"
            )
            for k, (_, kind) in METRIC_LABELS.items()
        }
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch", column_config=config
        )
        st.caption(
            "값을 계산할 수 없는 지표는 0으로 채우지 않고 비워 둡니다 (evaluation-xai 규칙)."
        )

    with tab_curve:
        empty_state(
            "학습 곡선은 rl-engine 실데이터 학습 결과를 기다리는 중입니다.",
            "PPO 학습(10만 스텝 이상)의 Episode Reward 추이가 연동되면 보상 함수 3종(단순 수익률·샤프·MDD 페널티) "
            "비교 곡선이 이곳에 표시됩니다.",
            icon=":material/model_training:",
        )

    limitations = bt.get("limitations") or []
    if limitations:
        with st.expander("이 결과의 한계", icon=":material/info:"):
            for item in limitations:
                st.markdown(f"- {item}")
