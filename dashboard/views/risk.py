"""리스크 모니터링: VaR/CVaR 게이지, MDD 추이, Safe-Guard 알림."""

import pandas as pd
import streamlit as st

from core import api, charts
from core.ui import empty_state, evaluated_runs, load, metric_value, page_header, pct, strategy_label, table_view

MDD_LIMIT = 0.15  # Safe-Guard: 낙폭 15% 초과 시 에피소드 조기 종료 / 위험자산 축소
VAR_LIMIT = 0.03  # 일간 VaR 주의 기준 (모니터링용 표시 기준)

bt = load(api.backtest)
page_header("리스크 모니터링", "손실 위험 지표와 Safe-Guard(MDD 15% 한도) 발동 현황",
            data_mode=bt.get("data_mode") if bt else None)

if bt:
    runs = evaluated_runs(bt)
    run_ids = [k for k, _ in runs]
    sel = st.selectbox("모니터링할 포트폴리오", run_ids, index=len(run_ids) - 1 if run_ids else 0,
                       format_func=lambda k: f"{strategy_label(bt['runs'][k]['strategy'])} · {bt['runs'][k]['fold_id']}")
    run = bt["runs"][sel]
    mdd = metric_value(run, "mdd") or 0.0
    var95 = metric_value(run, "var_95") or 0.0
    cvar95 = metric_value(run, "cvar_95") or 0.0

    if mdd >= MDD_LIMIT:
        st.error(f"**Safe-Guard 발동** · 최대 낙폭 {pct(-mdd, 2)}가 한도 -15%를 넘었습니다. 위험자산 비중을 축소합니다.",
                 icon=":material/gpp_bad:")
    elif mdd >= MDD_LIMIT * 0.6:
        st.warning(f"**주의** · 최대 낙폭 {pct(-mdd, 2)}로 Safe-Guard 한도(-15%)의 {mdd / MDD_LIMIT:.0%}에 도달했습니다.",
                   icon=":material/warning:")
    else:
        st.success(f"**정상 운용 중** · 최대 낙폭 {pct(-mdd, 2)} (한도 -15%의 {mdd / MDD_LIMIT:.0%})",
                   icon=":material/verified_user:")

    g1, g2, g3 = st.columns(3)
    for col, value, limit, title, key in [
        (g1, var95, VAR_LIMIT, "일간 VaR 95%", "g_var"),
        (g2, cvar95, VAR_LIMIT, "일간 CVaR 95%", "g_cvar"),
        (g3, mdd, MDD_LIMIT, "최대 낙폭 MDD", "g_mdd"),
    ]:
        with col, st.container(border=True):
            charts.show(charts.gauge(value, limit=limit, title=title), key=key)
            st.caption(f"기준선 {limit:.0%} · 현재 {value:.2%}")

    ts = load(api.timeseries, sel)
    if ts:
        with st.container(border=True):
            st.markdown("**낙폭 추이 (Underwater)**")
            st.caption("고점 대비 하락률. 빨간 선(-15%)을 넘으면 Safe-Guard가 발동합니다.")
            charts.show(charts.underwater(ts["dates"], ts["drawdown"], limit=MDD_LIMIT), key="risk_uw")
            frame = pd.DataFrame({"날짜": ts["dates"], "낙폭": [-d for d in ts["drawdown"]],
                                  "일간 수익률": ts["net_return"]})
            table_view(frame.sort_values("낙폭").head(10), label="표로 보기 (낙폭 상위 10일)",
                       column_config={"낙폭": st.column_config.NumberColumn(format="percent"),
                                      "일간 수익률": st.column_config.NumberColumn(format="percent")})

        with st.container(border=True):
            st.markdown("**Safe-Guard 발동 기록**")
            events = [(d, t, a) for d, t, a in zip(ts["dates"], ts["safeguard_triggered"], ts["guard_active"]) if t or a]
            if events:
                st.dataframe(pd.DataFrame(events, columns=["날짜", "발동", "보호 모드"]), hide_index=True, width="stretch")
            else:
                empty_state("이 구간에서는 발동 기록이 없습니다.", "낙폭이 한도 안에 머물러 정상 운용됐습니다.",
                            icon=":material/check_circle:")
