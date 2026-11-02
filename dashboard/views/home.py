"""개요: 핵심 지표와 현재 포트폴리오를 한 화면에."""

from collections import defaultdict

import pandas as pd
import streamlit as st

from core import api, charts
from core.assets import ASSET_CLASSES, DEFAULT_TICKERS, asset_class
from core.ui import (
    evaluated_runs,
    load,
    metric_value,
    page_header,
    pct,
    num,
    strategy_label,
    table_view,
)

bt = load(api.backtest)
page_header(
    "포트폴리오 개요",
    "강화학습 자산배분 · 에이전트 리서치 · 설명 가능성(XAI)을 한눈에",
    data_mode=bt.get("data_mode") if bt else None,
)

if bt:
    runs = evaluated_runs(bt)
    strategies = sorted({r["strategy"] for _, r in runs})
    folds = sorted({r["fold_id"] for _, r in runs})
    latest = folds[-1] if folds else None

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    strategy = c1.segmented_control(
        "전략",
        strategies,
        default=strategies[0] if strategies else None,
        format_func=strategy_label,
        required=True,
    )
    c2.caption(f"기준 구간: **{latest}** (Walk-Forward 테스트 1년)")

    run_id, run = next(
        (
            (k, r)
            for k, r in runs
            if r["strategy"] == strategy and r["fold_id"] == latest
        ),
        (None, None),
    )
    if run:
        ts = load(api.timeseries, run_id)
        spark = ts["cumulative_return"][::5] if ts else None
        dd = [-d for d in ts["drawdown"][::5]] if ts else None
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "누적 수익률",
            pct(metric_value(run, "cumulative_return"), 2),
            border=True,
            chart_data=spark,
            chart_type="area",
            help="테스트 구간 시작 대비 누적 수익률",
        )
        k2.metric(
            "샤프 비율",
            num(metric_value(run, "sharpe")),
            border=True,
            height="stretch",
            help="변동성 1단위당 초과수익. 높을수록 위험 대비 성과가 좋음",
        )
        k3.metric(
            "최대 낙폭 (MDD)",
            pct(-(metric_value(run, "mdd") or 0), 2),
            border=True,
            chart_data=dd,
            chart_type="area",
            help="고점 대비 최대 하락폭. Safe-Guard 한도는 -15%",
        )
        k4.metric(
            "VaR 95% (일간)",
            pct(-(metric_value(run, "var_95") or 0), 2),
            border=True,
            height="stretch",
            help="하루 동안 95% 확률로 이보다 크게 잃지 않는 손실 한도",
        )

    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True):
        st.markdown(f"**누적 수익률 비교 · {latest}**")
        palette_map = charts.color_map(strategies)
        series, rows = {}, []
        for k, r in runs:
            if r["fold_id"] != latest:
                continue
            data = load(api.timeseries, k)
            if data:
                name = strategy_label(r["strategy"])
                series[name] = (data["dates"], data["cumulative_return"])
                rows.append(
                    {"전략": name, "누적 수익률": data["cumulative_return"][-1]}
                )
        if series:
            colors = {strategy_label(s): palette_map[s] for s in strategies}
            charts.show(charts.line(series, colors=colors, height=320), key="home_line")
            table_view(
                pd.DataFrame(rows),
                column_config={
                    "누적 수익률": st.column_config.NumberColumn(format="percent")
                },
            )

    with right, st.container(border=True):
        st.markdown("**현재 추천 비중 · 자산군**")
        opt = load(
            api.optimize, tuple(DEFAULT_TICKERS), True, "2024-12-31T00:00:00-05:00"
        )
        if opt:
            by_class = defaultdict(float)
            for ticker, w in opt.get("weights", {}).items():
                by_class[asset_class(ticker)] += w
            labels = [c for c in ASSET_CLASSES if c in by_class]
            charts.show(
                charts.donut(
                    labels,
                    [by_class[c] for c in labels],
                    colors=charts.color_map(ASSET_CLASSES),
                    center="12개 자산",
                    height=290,
                ),
                key="home_donut",
            )
            if opt.get("is_mock") or opt.get("status") != "ok":
                st.caption(
                    ":material/info: 강화학습 모델 연동 전 시뮬레이션 비중입니다."
                )

st.markdown("#### 바로가기")
cols = st.columns(3)
links = [
    (
        "views/portfolio.py",
        "포트폴리오 현황",
        ":material/donut_large:",
        "종목별 비중 최적화와 비중 변화",
    ),
    (
        "views/risk.py",
        "리스크 모니터링",
        ":material/shield:",
        "VaR·CVaR 게이지, 낙폭, Safe-Guard",
    ),
    (
        "views/research.py",
        "에이전트 리서치",
        ":material/travel_explore:",
        "질문 → 추론 과정 → 출처 포함 리포트",
    ),
]
for col, (target, label, icon, desc) in zip(cols, links):
    with col, st.container(border=True):
        st.page_link(target, label=label, icon=icon)
        st.caption(desc)
