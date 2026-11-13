"""리스크 모니터링: VaR/CVaR 게이지, MDD 추이, Safe-Guard 알림 (현재 모델의 실제 백테스트).

아래쪽에는 리스크 태그 Safe-Guard(위험 이벤트 → 비중 축소)의 발동 현황을 보여 준다.
"""

import pandas as pd
import streamlit as st

from core import api, charts
from core.assets import asset_name
from core.ui import (
    empty_state,
    load,
    page_header,
    pct,
    perf_entries,
    risk_tag_table,
    table_view,
    window_label,
)

VAR_LIMIT = 0.03  # 일간 VaR 주의 기준 (모니터링용 표시 기준)

perf = load(api.performance)
page_header(
    "리스크 모니터링",
    "손실 위험 지표, MDD Safe-Guard(15% 한도), 리스크 태그 Safe-Guard 발동 현황",
    data_mode=perf.get("data_mode") if perf else None,
)

if perf:
    mdd_limit = perf["mdd_limit"]
    model = perf_entries(perf)["model"]
    options = ["all"] + [w["window"] for w in perf["windows"]]
    choice = st.segmented_control(
        "기간",
        options,
        default="all",
        format_func=lambda o: "4년 전체" if o == "all" else window_label(perf, o),
        required=True,
    )
    if choice == "all":
        metrics = model["total"]
        start, end = perf["windows"][0]["start"], perf["windows"][-1]["end"]
    else:
        metrics = next(w["metrics"] for w in model["windows"] if w["window"] == choice)
        period = next(w for w in perf["windows"] if w["window"] == choice)
        start, end = period["start"], period["end"]

    mdd = metrics.get("mdd") or 0.0
    var95 = metrics.get("var_95") or 0.0
    cvar95 = metrics.get("cvar_95") or 0.0

    if mdd >= mdd_limit:
        st.error(
            f"**Safe-Guard 한도 초과** · 최대 낙폭 {pct(-mdd, 1)}가 한도 "
            f"-{mdd_limit:.0%}를 넘었습니다. 학습 환경에서는 이 시점에 에피소드가 "
            "조기 종료되며, 아래 백테스트는 전략 비교를 위해 기간 끝까지 운용한 결과입니다.",
            icon=":material/gpp_bad:",
        )
    elif mdd >= mdd_limit * 0.6:
        st.warning(
            f"**주의** · 최대 낙폭 {pct(-mdd, 1)}로 "
            f"Safe-Guard 한도의 {mdd / mdd_limit:.0%}에 도달했습니다.",
            icon=":material/warning:",
        )
    else:
        st.success(
            f"**정상 운용** · 최대 낙폭 {pct(-mdd, 1)} "
            f"(한도 -{mdd_limit:.0%}의 {mdd / mdd_limit:.0%})",
            icon=":material/verified_user:",
        )

    g1, g2, g3 = st.columns(3)
    for col, value, limit, title, key in [
        (g1, var95, VAR_LIMIT, "일간 VaR 95%", "g_var"),
        (g2, cvar95, VAR_LIMIT, "일간 CVaR 95%", "g_cvar"),
        (g3, mdd, mdd_limit, "최대 낙폭 MDD", "g_mdd"),
    ]:
        with col, st.container(border=True):
            charts.show(charts.gauge(value, limit=limit, title=title), key=key)
            st.caption(f"기준선 {limit:.0%} · 현재 {value:.2%}")

    series = perf["series"]
    frame = pd.DataFrame(
        {
            "날짜": series["dates"],
            "낙폭": series["model"]["drawdown"],
            "누적 수익률": series["model"]["cumulative_return"],
        }
    )
    frame = frame[(frame["날짜"] >= start) & (frame["날짜"] <= end)]

    with st.container(border=True):
        st.markdown("**낙폭 추이 (Underwater)**")
        st.caption(
            "4년 연속 운용 기준 고점 대비 하락률. 빨간 선(-15%) 아래가 Safe-Guard 한도 초과 구간입니다."
        )
        charts.show(
            charts.underwater(
                frame["날짜"].tolist(), frame["낙폭"].tolist(), limit=mdd_limit
            ),
            key="risk_uw",
        )
        table_view(
            frame.assign(낙폭=-frame["낙폭"]).sort_values("낙폭").head(10),
            label="표로 보기 (낙폭 상위 10일)",
            column_config={
                "낙폭": st.column_config.NumberColumn(format="percent"),
                "누적 수익률": st.column_config.NumberColumn(format="percent"),
            },
        )

    with st.container(border=True):
        st.markdown("**Safe-Guard 한도 초과 기록**")
        events = [
            e for e in perf["breaches"] if e["end"] >= start and e["start"] <= end
        ]
        if events:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "시작": e["start"],
                            "종료": e["end"],
                            "거래일 수": e["days"],
                            "최대 낙폭": -e["max_drawdown"],
                        }
                        for e in events
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "최대 낙폭": st.column_config.NumberColumn(format="percent")
                },
            )
        else:
            empty_state(
                "이 기간에는 한도 초과 기록이 없습니다.",
                "낙폭이 한도 안에 머물렀습니다.",
                icon=":material/check_circle:",
            )

# ---------------- 리스크 태그 Safe-Guard ----------------
st.markdown("#### 리스크 태그 Safe-Guard")
guard = load(api.risk_tags)
if guard:
    rule, days = guard["rule"], guard["days"]
    st.caption(
        "리서치 에이전트가 탐지한 위험 이벤트(급등락·실적 쇼크 등)를 리스크 태그로 받아, "
        f"위험 점수(심각도 × 신뢰도)가 {rule['threshold']:.2f} 이상인 자산의 비중을 "
        f"(점수 × {rule['max_cut']:.0%})만큼 자동으로 줄입니다. "
        f"대상 기간은 현재 모델의 평가 구간 {guard['period']['start']} ~ "
        f"{guard['period']['end']}입니다."
    )
    applied_tags = sum(1 for t in guard["tags"] if t["applied"])
    peak = max(days, key=lambda d: d["moved_weight"]) if days else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "발동일",
        f"{guard['adjusted_days']}일",
        f"전체 {guard['trading_days']}거래일",
        delta_color="off",
        delta_arrow="off",
        border=True,
        help="위험 신호 때문에 비중을 실제로 조정한 거래일 수",
    )
    m2.metric(
        "발동일 평균 축소 비중",
        f"{guard['average_moved'] * 100:.1f}%p",
        border=True,
        height="stretch",
        help="발동한 날 위험 자산에서 덜어 낸 비중 합계의 평균",
    )
    m3.metric(
        "최대 축소 비중",
        f"{guard['max_moved'] * 100:.1f}%p",
        peak["date"] if peak else None,
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    m4.metric(
        "반영된 태그",
        f"{applied_tags}건",
        f"기간 내 유효 {len(guard['tags'])}건",
        delta_color="off",
        delta_arrow="off",
        border=True,
        help="기준 점수 이상이라 비중 축소에 쓰인 태그 수",
    )

    if days:
        left, right = st.columns([3, 2], gap="medium")
        with left, st.container(border=True):
            st.markdown("**발동일별 축소 비중**")
            st.caption("막대가 있는 날이 Safe-Guard가 비중을 조정한 결정일입니다.")
            charts.show(
                charts.date_bars(
                    [d["date"] for d in days],
                    [d["moved_weight"] for d in days],
                    name="축소 비중",
                ),
                key="guard_days",
            )
            table_view(
                pd.DataFrame(
                    [
                        {
                            "결정일": d["date"],
                            "축소 비중": d["moved_weight"],
                            "대상 자산": ", ".join(asset_name(a) for a in d["assets"]),
                        }
                        for d in sorted(days, key=lambda d: -d["moved_weight"])
                    ]
                ),
                label="표로 보기 (축소 비중 큰 순)",
                column_config={
                    "축소 비중": st.column_config.NumberColumn(format="percent")
                },
            )
        with right, st.container(border=True):
            st.markdown("**자산별 발동일 수**")
            st.caption("위험 신호로 비중이 줄어든 날이 많은 자산 순")
            assets = guard["by_asset"]
            charts.show(
                charts.hbar(
                    [asset_name(a["asset"]) for a in assets],
                    [a["days"] for a in assets],
                    percent=False,
                    digits=0,
                ),
                key="guard_assets",
            )
            table_view(
                pd.DataFrame(
                    [
                        {
                            "자산": asset_name(a["asset"]),
                            "발동일 수": a["days"],
                            "하루 평균 축소": a["moved_weight"],
                        }
                        for a in assets
                    ]
                ),
                column_config={
                    "하루 평균 축소": st.column_config.NumberColumn(format="percent")
                },
            )
    else:
        empty_state(
            "이 구간에는 Safe-Guard가 발동한 날이 없습니다.",
            "기준 점수 이상인 위험 이벤트가 없었습니다.",
            icon=":material/check_circle:",
        )

    if guard["tags"]:
        with st.expander(
            f"기간 내 리스크 태그 {len(guard['tags'])}건 (출처 포함)",
            icon=":material/crisis_alert:",
        ):
            risk_tag_table(guard["tags"])
    st.caption(f":material/info: {guard['limitation']}")
