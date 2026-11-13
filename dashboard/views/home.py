"""개요: 현재 모델의 핵심 성과와 추천 비중을 한 화면에."""

from collections import defaultdict

import pandas as pd
import streamlit as st

from core import api, charts
from core.assets import ASSET_CLASSES, DEFAULT_TICKERS, asset_class
from core.ui import (
    comparison_series,
    load,
    num,
    page_header,
    pct,
    perf_entries,
    table_view,
)

perf = load(api.performance)
page_header(
    "포트폴리오 개요",
    "강화학습 자산배분 · 에이전트 리서치 · 설명 가능성(XAI)을 한눈에",
    data_mode=perf.get("data_mode") if perf else None,
)


def delta(value: float | None, base: float | None, *, percent: bool) -> str | None:
    """동일가중 대비 차이 문구 (비교할 수 없으면 표시하지 않음)."""
    if value is None or base is None:
        return None
    diff = value - base
    return f"{diff * 100:+.1f}%p" if percent else f"{diff:+.2f}"


if perf:
    entries = perf_entries(perf)
    model, base = entries["model"]["total"], entries["baseline"]["total"]
    series = perf["series"]
    period = f"{perf['windows'][0]['start']} ~ {perf['windows'][-1]['end']}"
    st.caption(
        f"현재 모델 `{perf['model_name']}` · Walk-Forward 테스트 {period} "
        "(구간마다 재학습, 거래비용 차감 후) · 아래 작은 숫자는 동일가중 대비 차이"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "누적 수익률 (4년)",
        pct(model["cumulative_return"], 1),
        delta(model["cumulative_return"], base["cumulative_return"], percent=True),
        border=True,
        chart_data=series["model"]["cumulative_return"][::5],
        chart_type="area",
        help="테스트 구간 4년을 이어 붙인 out-of-sample 누적 수익률",
    )
    k2.metric(
        "샤프 비율",
        num(model["sharpe"]),
        delta(model["sharpe"], base["sharpe"], percent=False),
        border=True,
        height="stretch",
        help="변동성 1단위당 수익. 높을수록 위험 대비 성과가 좋음",
    )
    k3.metric(
        "최대 낙폭 (MDD)",
        pct(-(model["mdd"] or 0), 1),
        delta(model["mdd"], base["mdd"], percent=True),
        delta_color="inverse",
        border=True,
        chart_data=[-d for d in series["model"]["drawdown"][::5]],
        chart_type="area",
        help="고점 대비 최대 하락폭. Safe-Guard 한도는 -15%",
    )
    k4.metric(
        "VaR 95% (일간)",
        pct(-(model["var_95"] or 0), 2),
        delta(model["var_95"], base["var_95"], percent=True),
        delta_color="inverse",
        border=True,
        height="stretch",
        help="하루 동안 95% 확률로 이보다 크게 잃지 않는 손실 한도",
    )

left, right = st.columns([3, 2], gap="medium")
if perf:
    with left, st.container(border=True):
        st.markdown("**누적 수익률 비교 · 4년 연결**")
        lines = comparison_series(perf, "cumulative_return")
        colors = charts.color_map(list(lines))
        charts.show(charts.line(lines, colors=colors, height=320), key="home_line")
        table_view(
            pd.DataFrame(
                [{"전략": name, "누적 수익률": y[-1]} for name, (_, y) in lines.items()]
            ),
            column_config={
                "누적 수익률": st.column_config.NumberColumn(format="percent")
            },
        )

with right, st.container(border=True):
    st.markdown("**현재 추천 비중 · 자산군**")
    opt = load(api.optimize, tuple(DEFAULT_TICKERS), True, None)
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
        if opt.get("status") != "ok":
            st.caption(":material/info: RL 서버 장애로 동일 비중을 표시 중입니다.")
        elif opt.get("decision_date"):
            note = f"{opt['decision_date']} 장 마감 후 결정"
            if opt.get("risk_override_applied"):
                cut = len(opt["risk_guard"]["adjustments"])
                note += f" · 리스크 태그 Safe-Guard가 {cut}개 자산 비중 축소"
            st.caption(note)

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
        "VaR·CVaR 게이지, 낙폭, Safe-Guard 발동 현황",
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
