"""화면 공통 부품: 페이지 머리말, 상태 배지, 오류·빈 상태 카드, 숫자 형식, 면책 조항."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from core.api import ApiError

DISCLAIMER = (
    "본 시스템은 교육 목적으로 개발되었으며 실제 투자 조언이나 자문에 사용할 수 없습니다. "
    "백테스팅 및 시뮬레이션 성과는 미래의 수익을 보장하지 않습니다."
)

SERVICE_LABELS = {
    "evaluation_xai": ("평가·XAI", "백테스트 · SHAP · ANOVA"),
    "agentic_research": ("에이전트 리서치", "LangGraph RAG"),
    "rl_engine": ("강화학습 엔진", "PPO 포트폴리오"),
}

METRIC_LABELS = {
    "cumulative_return": ("누적 수익률", "pct"),
    "cagr": ("CAGR", "pct"),
    "annualized_volatility": ("연환산 변동성", "pct"),
    "var_95": ("VaR 95%", "pct"),
    "cvar_95": ("CVaR 95%", "pct"),
    "mdd": ("MDD", "pct"),
    "sharpe": ("샤프", "num"),
    "sortino": ("소르티노", "num"),
    "calmar": ("칼마", "num"),
    "alpha": ("알파", "pct"),
    "beta": ("베타", "num"),
    "information_ratio": ("정보비율", "num"),
}

STRATEGY_LABELS = {
    "equal_weight": "동일가중",
    "mvo": "MVO (평균-분산)",
    "ppo": "PPO 강화학습",
}


def page_header(title: str, subtitle: str, *, data_mode: str | None = None) -> None:
    left, right = st.columns([5, 2], vertical_alignment="bottom")
    with left:
        st.title(title)
        st.caption(subtitle)
    with right:
        with st.container(horizontal=True, horizontal_alignment="right"):
            if data_mode == "synthetic":
                st.badge(
                    "DEMO · 합성 데이터",
                    icon=":material/science:",
                    color="orange",
                    help="실제 시장 데이터·학습된 PPO 모델 결과가 아닌 평가 파이프라인 검증용 합성 데이터입니다.",
                )
            elif data_mode == "real":
                st.badge("실데이터", icon=":material/verified:", color="green")


def load(fn: Callable[..., Any], *args, spinner: str = "불러오는 중…") -> Any | None:
    """API 호출을 감싸 실패 시 안내 카드를 보여주고 None을 돌려준다."""
    try:
        with st.spinner(spinner):
            return fn(*args)
    except ApiError as error:
        error_card(error)
        return None


def error_card(error: ApiError) -> None:
    with st.container(border=True):
        st.markdown(f"**:material/cloud_off: {error.message}**")
        if error.hint:
            st.caption(error.hint)


def empty_state(
    title: str, body: str, icon: str = ":material/hourglass_empty:"
) -> None:
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**")
        st.caption(body)


def table_view(df: pd.DataFrame, *, label: str = "표로 보기", **kwargs) -> None:
    """차트와 같은 값을 표로도 제공 (색만으로 값을 읽지 않도록)."""
    with st.expander(label, icon=":material/table_view:"):
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def pct(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}"


def num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def metric_value(run: dict, key: str) -> float | None:
    """evaluation-xai 지표는 {value, status, reason} 형태. 계산 불가 값은 None으로 둔다(0으로 바꾸지 않음)."""
    item = run.get("metrics", {}).get(key)
    if isinstance(item, dict):
        return item.get("value")
    return item


def strategy_label(strategy: str | None) -> str:
    return STRATEGY_LABELS.get(strategy or "", strategy or "-")


def run_label(bt: dict, run_id: str) -> str:
    """백테스트 실행 ID를 '전략 · 구간' 형태로 표시."""
    run = bt["runs"][run_id]
    return f"{strategy_label(run.get('strategy'))} · {run.get('fold_id', '-')}"


def evaluated_runs(bt: dict) -> list[tuple[str, dict]]:
    """지표가 계산된 백테스트 실행만 (전략, 구간) 순으로."""
    runs = [(k, v) for k, v in bt.get("runs", {}).items() if v.get("metrics")]
    return sorted(
        runs, key=lambda kv: (kv[1].get("strategy", ""), kv[1].get("fold_id", ""))
    )


def footer() -> None:
    st.divider()
    st.caption(f":material/gavel: **금융 면책 조항** · {DISCLAIMER}")
