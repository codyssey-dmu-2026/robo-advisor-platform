"""포트폴리오 현황: 현재 자산 비중(POST /optimize)과 누적 수익률·비중 변화(백테스트 시계열)."""

from collections import defaultdict
from datetime import date, datetime, time, timezone

import pandas as pd
import streamlit as st

from core import api, charts
from core.assets import ASSET_CLASSES, DEFAULT_TICKERS, asset_class, asset_name
from core.ui import (
    evaluated_runs,
    load,
    page_header,
    run_label,
    strategy_label,
    table_view,
)

bt = load(api.backtest)
page_header(
    "포트폴리오 현황",
    "강화학습 엔진이 제안하는 현재 자산 비중과 과거 성과 추이",
    data_mode=bt.get("data_mode") if bt else None,
)

# ---------------- 현재 최적 비중 ----------------
with st.container(border=True):
    c1, c2, c3 = st.columns([3.6, 1.4, 1.6], vertical_alignment="bottom")
    tickers = c1.multiselect(
        "투자 대상 자산",
        DEFAULT_TICKERS,
        default=DEFAULT_TICKERS,
        format_func=asset_name,
        placeholder="자산을 선택하세요",
    )
    as_of = c2.date_input(
        "기준일",
        value=date(2024, 12, 31),
        help="이 날짜까지의 정보로만 판단 (미래 정보 누수 방지)",
    )
    risk_override = c3.toggle(
        "리스크 태그 반영",
        value=True,
        help="에이전트 리서치가 탐지한 위험 이벤트를 비중에 반영",
    )

if not tickers:
    st.info("자산을 1개 이상 선택하세요.", icon=":material/touch_app:")
else:
    as_of_iso = datetime.combine(as_of, time(0), tzinfo=timezone.utc).isoformat()
    opt = load(
        api.optimize,
        tuple(sorted(tickers)),
        risk_override,
        as_of_iso,
        spinner="포트폴리오 최적화 중…",
    )
    if opt:
        weights = opt.get("weights", {})
        if opt.get("is_mock") or opt.get("status") != "ok":
            st.warning(
                "실제 PPO 추론 결과가 아닌 **인터페이스 검증용 시뮬레이션 비중**입니다. "
                "rl-engine 모델 연동 후 실제 값으로 바뀝니다.",
                icon=":material/science:",
            )
        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("편입 자산", f"{len(weights)}개", border=True)
        k2.metric(
            "최대 비중",
            f"{ranked[0][1]:.1%}",
            help=asset_name(ranked[0][0]),
            border=True,
        )
        k3.metric(
            "거래비용 (편도)",
            "0.065%",
            border=True,
            help="수수료 0.015% + 슬리피지 0.05%, 비중 변화량에 비례해 차감",
        )
        k4.metric(
            "Safe-Guard",
            "발동" if opt.get("safeguard_triggered") else "정상",
            border=True,
            help="MDD 15% 초과 시 위험자산 비중 축소",
        )

        left, right = st.columns([3, 2], gap="medium")
        with left, st.container(border=True):
            st.markdown("**종목별 비중**")
            charts.show(
                charts.hbar([asset_name(t) for t, _ in ranked], [w for _, w in ranked]),
                key="pf_bar",
            )
        with right, st.container(border=True):
            st.markdown("**자산군 구성**")
            by_class = defaultdict(float)
            for t, w in weights.items():
                by_class[asset_class(t)] += w
            labels = [c for c in ASSET_CLASSES if c in by_class]
            charts.show(
                charts.donut(
                    labels,
                    [by_class[c] for c in labels],
                    colors=charts.color_map(ASSET_CLASSES),
                    center=f"{len(weights)}개 자산",
                ),
                key="pf_donut",
            )
        table_view(
            pd.DataFrame(
                [
                    {"자산": asset_name(t), "자산군": asset_class(t), "비중": w}
                    for t, w in ranked
                ]
            ),
            column_config={
                "비중": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1
                )
            },
        )
        st.caption(
            f"모델: `{opt.get('model_used', '-')}` · 기준 시각 {opt.get('as_of', '-')}"
        )

# ---------------- 백테스트 성과 추이 ----------------
st.markdown("#### 성과 추이 (Walk-Forward 백테스트)")
if bt:
    runs = evaluated_runs(bt)
    strategies = sorted({r["strategy"] for _, r in runs})
    colors = {strategy_label(s): c for s, c in charts.color_map(strategies).items()}

    with st.container(border=True):
        st.markdown("**누적 수익률 · 전체 테스트 구간 연결**")
        st.caption(
            "구간마다 재학습한 결과를 이어 붙인 out-of-sample 성과 (각 구간 시작 시 자본 연속)"
        )
        series, rows = {}, []
        for s in strategies:
            dates, values, growth = [], [], 1.0
            for run_id, r in runs:
                if r["strategy"] != s:
                    continue
                ts = load(api.timeseries, run_id)
                if not ts:
                    continue
                prev = growth
                for d, cr in zip(ts["dates"], ts["cumulative_return"]):
                    dates.append(d)
                    values.append(prev * (1 + cr) - 1)
                growth = prev * (1 + ts["cumulative_return"][-1])
            if dates:
                series[strategy_label(s)] = (dates, values)
                rows.append(
                    {
                        "전략": strategy_label(s),
                        "기간": f"{dates[0]} ~ {dates[-1]}",
                        "누적 수익률": values[-1],
                    }
                )
        if series:
            charts.show(charts.line(series, colors=colors, height=360), key="pf_cum")
            table_view(
                pd.DataFrame(rows),
                column_config={
                    "누적 수익률": st.column_config.NumberColumn(format="percent")
                },
            )
        st.caption(
            ":material/info: 벤치마크(S&P500·KOSPI) 비교는 실데이터 평가 결과 연동 후 표시됩니다."
        )

    with st.container(border=True):
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.markdown("**비중 변화**")
        run_ids = [k for k, _ in runs]
        sel = c2.selectbox(
            "백테스트 실행",
            run_ids,
            index=len(run_ids) - 1 if run_ids else 0,
            format_func=lambda k: run_label(bt, k),
            label_visibility="collapsed",
        )
        ts = load(api.timeseries, sel) if sel else None
        if ts and ts["weights"]:
            # 종목이 많으면 색이 8개를 넘으므로 자산군으로 묶고, 자산군 정보가 없으면 상위 5개 + 기타
            frame = pd.DataFrame(ts["weights"], index=ts["dates"]).fillna(0)
            classes = {c: asset_class(c) for c in frame.columns}
            if set(classes.values()) <= set(ASSET_CLASSES) | {"현금"}:
                grouped = frame.T.groupby(classes).sum().T
            else:
                top = (
                    frame.drop(columns=["CASH"], errors="ignore")
                    .mean()
                    .nlargest(5)
                    .index.tolist()
                )
                grouped = frame[top].copy()
                grouped["기타·현금"] = frame.drop(columns=top).sum(axis=1)
            order = list(grouped.columns)
            charts.show(
                charts.stacked_area(
                    list(grouped.index),
                    {c: grouped[c].tolist() for c in order},
                    colors=charts.color_map(order),
                ),
                key="pf_area",
            )
            table_view(
                grouped.reset_index(names="날짜").tail(20),
                label="표로 보기 (최근 20일)",
                column_config={
                    c: st.column_config.NumberColumn(format="percent") for c in order
                },
            )
