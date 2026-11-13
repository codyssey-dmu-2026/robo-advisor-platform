"""포트폴리오 현황: 현재 자산 비중(POST /optimize)과 누적 수익률·비중 변화(GET /performance)."""

from collections import defaultdict
from datetime import date, datetime, time, timezone

import pandas as pd
import streamlit as st

from core import api, charts
from core.assets import ASSET_CLASSES, DEFAULT_TICKERS, asset_class, asset_name
from core.ui import (
    comparison_series,
    empty_state,
    load,
    page_header,
    risk_tag_table,
    table_view,
)

perf = load(api.performance)
page_header(
    "포트폴리오 현황",
    "강화학습 엔진이 제안하는 현재 자산 비중과 과거 성과 추이",
    data_mode=perf.get("data_mode") if perf else None,
)

# ---------------- 현재 최적 비중 ----------------
# 선택 가능한 기준일은 현재 모델의 평가 구간을 RL 서버에서 받아 온다
try:
    rl_status = api.health().get("services", {}).get("rl_engine", {})
except api.ApiError:
    rl_status = {}
earliest = date.fromisoformat(rl_status.get("as_of_min", "2025-01-03"))
latest = date.fromisoformat(rl_status.get("as_of_max", "2026-01-01"))

# 리스크 태그 Safe-Guard가 작동한 날 목록 (없어도 화면은 그대로 동작)
try:
    guard_days = api.risk_tags().get("days", [])
except api.ApiError:
    guard_days = []
guard_days = [
    d for d in guard_days if earliest <= date.fromisoformat(d["as_of"]) <= latest
]
moved_on = {d["as_of"]: d for d in guard_days}

# 기준일은 달력과 '발동일 바로가기' 양쪽에서 바꿀 수 있어 세션 상태로 관리한다
picked = st.session_state.get("pf_as_of", latest)
st.session_state["pf_as_of"] = min(max(picked, earliest), latest)


def _jump_to_guard_day() -> None:
    if st.session_state.get("pf_jump"):
        st.session_state["pf_as_of"] = date.fromisoformat(st.session_state["pf_jump"])


def _clear_jump() -> None:
    st.session_state["pf_jump"] = None


with st.container(border=True):
    c1, c2, c3, c4 = st.columns([2.2, 1.3, 2.4, 1.5], vertical_alignment="bottom")
    c1.markdown(f"**투자 대상** · 학습한 {len(DEFAULT_TICKERS)}개 자산 고정")
    as_of = c2.date_input(
        "기준일",
        min_value=earliest,
        max_value=latest,
        key="pf_as_of",
        on_change=_clear_jump,
        help="이 날짜 전에 끝난 거래일까지의 정보로만 판단 (미래 정보 누수 방지)",
    )
    c3.selectbox(
        "리스크 태그 발동일로 이동",
        [d["as_of"] for d in reversed(guard_days)],
        index=None,
        key="pf_jump",
        on_change=_jump_to_guard_day,
        format_func=lambda a: (
            f"{a} · {moved_on[a]['moved_weight']:.1%}p 축소 · "
            f"{len(moved_on[a]['assets'])}개 자산"
        ),
        placeholder=f"발동일 {len(guard_days)}일" if guard_days else "발동일 없음",
        disabled=not guard_days,
        help="리스크 태그 Safe-Guard가 비중을 조정한 기준일만 모아 둔 목록",
    )
    risk_override = c4.toggle(
        "리스크 태그 반영",
        value=True,
        help="에이전트가 탐지한 위험 신호가 강한 자산의 비중을 자동으로 줄입니다 "
        "(리스크 태그 Safe-Guard). 끄면 모델 원래 비중을 봅니다",
    )

as_of_iso = datetime.combine(as_of, time(0), tzinfo=timezone.utc).isoformat()
opt = load(
    api.optimize,
    tuple(DEFAULT_TICKERS),
    risk_override,
    as_of_iso,
    spinner="포트폴리오 최적화 중…",
)
if opt:
    weights = opt.get("weights", {})
    if opt.get("status") != "ok":
        st.warning(
            "RL 서버를 사용할 수 없어 **임시 동일 비중**을 표시합니다. "
            "실제 모델 결과가 아닙니다.",
            icon=":material/science:",
        )
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    guard = opt.get("risk_guard") or {}
    adjustments = guard.get("adjustments", [])
    live_tags = guard.get("tags", [])
    k1, k2, k3, k4 = st.columns(4)
    if not guard.get("available"):
        k1.metric(
            "리스크 태그 Safe-Guard",
            "—",
            border=True,
            height="stretch",
            help=guard.get("reason") or "리스크 태그를 불러오지 못해 모델 비중 그대로",
        )
    else:
        if adjustments:
            headline = f"{len(adjustments)}개 축소"
            detail = f"비중 {guard['moved_weight'] * 100:.1f}%p 이동"
        else:
            headline = "조정 없음" if guard.get("requested") else "꺼짐"
            detail = f"유효 태그 {len(live_tags)}건"
        k1.metric(
            "리스크 태그 Safe-Guard",
            headline,
            detail,
            delta_color="off",
            delta_arrow="off",
            border=True,
            height="stretch",
            help="결정일에 유효한 위험 이벤트의 점수(심각도 × 신뢰도)가 "
            f"{guard['rule']['threshold']:.2f} 이상이면 그 자산 비중을 줄입니다",
        )
    k2.metric(
        "최대 비중",
        f"{ranked[0][1]:.1%}",
        help=asset_name(ranked[0][0]),
        border=True,
        height="stretch",
    )
    k3.metric(
        "거래비용 (편도)",
        "0.065%",
        border=True,
        height="stretch",
        help="수수료 0.015% + 슬리피지 0.05%, 비중 변화량에 비례해 차감",
    )
    # 기준일(결정일) 당시 4년 연속 운용 기준 낙폭으로 Safe-Guard 한도 초과 여부 판정
    drawdown = None
    if perf and opt.get("decision_date"):
        series = perf["series"]
        past = [
            dd
            for d, dd in zip(series["dates"], series["model"]["drawdown"])
            if d <= opt["decision_date"]
        ]
        drawdown = past[-1] if past else None
    if drawdown is None:
        k4.metric("MDD Safe-Guard", "—", border=True, help="낙폭 정보를 불러오지 못함")
    else:
        over = drawdown > perf["mdd_limit"]
        k4.metric(
            "MDD Safe-Guard",
            "한도 초과" if over else "정상",
            f"낙폭 {-drawdown:.1%}",
            delta_color="inverse" if over else "off",
            delta_arrow="off",
            border=True,
            help="결정일 기준 고점 대비 낙폭. 15%를 넘으면 한도 초과",
        )

    # ---------------- 리스크 태그 Safe-Guard 조정 내역 ----------------
    if adjustments:
        rule = guard["rule"]
        with st.container(border=True):
            st.markdown("**리스크 태그 Safe-Guard 조정 내역**")
            st.caption(
                f"{opt.get('decision_date', '-')} 장 마감 기준으로 유효한 위험 이벤트를 "
                f"반영했습니다. 위험 점수(심각도 × 신뢰도)가 {rule['threshold']:.2f} "
                f"이상인 자산은 비중을 (점수 × {rule['max_cut']:.0%})만큼 줄이고, 줄인 "
                f"비중은 신호가 없는 자산에 원래 비율대로 나눕니다 "
                f"(자산당 상한 {rule['max_weight']:.0%})."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "자산": asset_name(a["asset"]),
                            "위험 점수": a["score"],
                            "모델 비중": a["before"],
                            "조정 후": a["after"],
                            "변화 (%p)": (a["after"] - a["before"]) * 100,
                        }
                        for a in adjustments
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "위험 점수": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.2f"
                    ),
                    "모델 비중": st.column_config.NumberColumn(format="percent"),
                    "조정 후": st.column_config.NumberColumn(format="percent"),
                    "변화 (%p)": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
            with st.expander(
                f"근거가 된 리스크 태그 {len(live_tags)}건", icon=":material/link:"
            ):
                risk_tag_table(live_tags)
            st.caption(
                ":material/info: 서빙 단계에서 적용하는 규칙입니다. 아래 백테스트 "
                "성과는 이 조정 없이 모델 원래 비중으로 계산한 값입니다."
            )
    elif live_tags:
        with st.container(border=True):
            st.markdown("**기준일에 유효한 리스크 태그**")
            st.caption(
                "리스크 태그 반영이 꺼져 있어 모델 원래 비중을 표시합니다."
                if not guard.get("requested")
                else "신호가 기준보다 약해 비중은 조정하지 않았습니다."
            )
            risk_tag_table(live_tags)

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
    model_weights = opt.get("model_weights") or {}
    table_view(
        pd.DataFrame(
            [
                {"자산": asset_name(t), "자산군": asset_class(t), "비중": w}
                # Safe-Guard가 비중을 바꾼 날에는 모델 원래 비중을 나란히 보여 준다
                | ({"모델 비중": model_weights.get(t)} if adjustments else {})
                for t, w in ranked
            ]
        ),
        column_config={
            "비중": st.column_config.ProgressColumn(
                format="percent", min_value=0, max_value=1
            ),
            "모델 비중": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        f"모델: `{opt.get('model_used', '-')}` · 기준 시각 {opt.get('as_of', '-')}"
    )

# ---------------- 백테스트 성과 추이 ----------------
st.markdown("#### 성과 추이 (Walk-Forward 백테스트)")
if perf:
    with st.container(border=True):
        st.markdown("**누적 수익률 · 전체 테스트 구간 연결**")
        st.caption(
            "구간(1년)마다 그 이전 데이터로만 재학습한 모델의 out-of-sample 성과를 "
            "이어 붙임 · 거래비용 차감 후 · 벤치마크는 S&P500 추종 ETF(SPY)"
        )
        lines = comparison_series(perf, "cumulative_return")
        colors = charts.color_map(list(lines))
        charts.show(charts.line(lines, colors=colors, height=360), key="pf_cum")
        dates = perf["series"]["dates"]
        table_view(
            pd.DataFrame(
                [
                    {
                        "전략": name,
                        "기간": f"{dates[0]} ~ {dates[-1]}",
                        "누적 수익률": y[-1],
                    }
                    for name, (_, y) in lines.items()
                ]
            ),
            column_config={
                "누적 수익률": st.column_config.NumberColumn(format="percent")
            },
        )

    with st.container(border=True):
        st.markdown("**비중 변화 · 현재 모델의 테스트 구간**")
        weights = perf.get("weights")
        if weights:
            frame = pd.DataFrame(
                weights["values"], index=weights["dates"], columns=weights["tickers"]
            )
            # 12개 종목은 색이 너무 많아 자산군 5개로 묶어 표시
            grouped = (
                frame.T.groupby({t: asset_class(t) for t in frame.columns}).sum().T
            )
            order = [c for c in ASSET_CLASSES if c in grouped.columns]
            st.caption(
                f"{weights['dates'][0]} ~ {weights['dates'][-1]} 매 거래일 장 마감 후 "
                "모델이 정한 목표 비중 (자산군 합계)"
            )
            charts.show(
                charts.stacked_area(
                    list(grouped.index),
                    {c: grouped[c].tolist() for c in order},
                    colors=charts.color_map(order),
                ),
                key="pf_area",
            )
            table_view(
                grouped[order].reset_index(names="날짜").tail(20),
                label="표로 보기 (최근 20일)",
                column_config={
                    c: st.column_config.NumberColumn(format="percent") for c in order
                },
            )
        else:
            empty_state(
                "RL 서버에서 모델 비중을 불러오지 못했습니다.",
                "모델이 로드되면 날짜별 비중 변화가 표시됩니다.",
                icon=":material/cloud_off:",
            )
