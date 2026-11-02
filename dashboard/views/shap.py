"""SHAP 해석: 특정 의사결정에서 어떤 피처 그룹이 자산 비중을 올리고 내렸는가."""

import pandas as pd
import streamlit as st

from core import api, charts
from core.assets import asset_name
from core.ui import empty_state, load, page_header, table_view

GROUP_LABELS = {
    "returns": "과거 수익률",
    "indicators": "기술적 지표 (RSI·MACD)",
    "risk_inputs": "리스크 입력 (낙폭·리스크 태그)",
}

bt = load(api.backtest)
page_header(
    "SHAP 의사결정 해석",
    '"왜 이 비중을 정했는가"를 피처 그룹별 기여도로 설명 (금융 XAI 대응)',
    data_mode=bt.get("data_mode") if bt else None,
)

explanations = (bt or {}).get("explanations", {})
if bt and not explanations:
    empty_state(
        "계산된 SHAP 설명이 없습니다.",
        "evaluation-xai 산출물에 explanations가 생성되면 표시됩니다.",
    )
elif explanations:
    keys = sorted(explanations)
    run_ids = sorted({k.split("/")[0] for k in keys})
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        run_id = c1.selectbox("실행 (run)", run_ids)
        decisions = [k.split("/")[1] for k in keys if k.startswith(f"{run_id}/")]
        decision_id = c2.selectbox("의사결정 시점 (decision)", decisions)
        assets = explanations[f"{run_id}/{decision_id}"].get("asset_names", [])
        target = c3.selectbox("대상 자산", assets, format_func=asset_name)

    exp = load(api.explain, run_id, decision_id, target, spinner="SHAP 값 조회 중…")
    if exp:
        groups = exp.get("groups", [])
        contributions = exp.get("contributions", [])
        base, predicted = exp.get("base_value", 0.0), exp.get("predicted_weight", 0.0)
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "기준 비중 (평균)",
            f"{base:.2%}",
            border=True,
            help="배경 데이터에서 이 자산의 평균 비중",
        )
        k2.metric(
            "예측 비중",
            f"{predicted:.2%}",
            delta=f"{predicted - base:+.2%}",
            border=True,
        )
        k3.metric(
            "SHAP 기여도 합계",
            f"{sum(contributions):+.2%}",
            border=True,
            help="기여도의 합 = 예측 비중 − 기준 비중 (가법성)",
        )
        st.caption(
            f"해석 방법 `{exp.get('method', '-')}` · 단계 `{exp.get('stage', '-')}`"
        )

        if groups and len(groups) == len(contributions):
            labels = [GROUP_LABELS.get(g, g) for g in groups]
            with st.container(border=True):
                st.markdown(f"**{asset_name(target)} 비중 결정 요인**")
                st.caption("파란 막대는 비중을 올린 요인, 빨간 막대는 내린 요인입니다.")
                charts.show(
                    charts.waterfall(base, labels, contributions), key="shap_wf"
                )
            top = max(zip(labels, contributions), key=lambda lc: abs(lc[1]))
            st.info(
                f"가장 큰 영향: **{top[0]}** ({top[1]:+.2%}p). "
                + (exp.get("interpretation") or ""),
                icon=":material/lightbulb:",
            )
            table_view(
                pd.DataFrame({"피처 그룹": labels, "SHAP 기여도": contributions}),
                column_config={
                    "SHAP 기여도": st.column_config.NumberColumn(format="%.4f")
                },
            )
        else:
            st.warning(
                "응답에 groups/contributions가 없거나 길이가 맞지 않습니다.",
                icon=":material/warning:",
            )

        with st.expander("규제 대응 메모", icon=":material/policy:"):
            st.markdown(
                "- 모든 추천 비중은 SHAP 기여도로 **결정 근거를 추적**할 수 있습니다.\n"
                "- 기여도의 합 = 예측 비중 − 기준 비중 (가법성). "
                "Safe-Guard 개입 이전의 정책 비중을 설명합니다.\n"
                "- Summary Plot·Force Plot 이미지는 evaluation-xai 산출물(`plots/`)에서 "
                "확인할 수 있습니다."
            )
