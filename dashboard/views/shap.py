"""SHAP 해석: 특정 의사결정에서 어떤 피처 그룹이 자산 비중을 올리고 내렸는가."""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core import api, charts
from core.api import ApiError
from core.assets import asset_name
from core.ui import empty_state, load, page_header, table_view

GROUP_LABELS = {
    "returns": "과거 수익률",
    "indicators": "기술적 지표 (RSI·MACD)",
    "risk_inputs": "리스크 입력 (낙폭·리스크 태그)",
}

catalog = load(api.explain_catalog)
data_mode = catalog.get("data_mode") if catalog else None
# 합성 데모 설명은 evaluation-xai가 명시적으로 허용해야만 돌려준다 (실제 의사결정 설명과 구분)
synthetic = data_mode == "synthetic"
page_header(
    "SHAP 의사결정 해석",
    '"왜 이 비중을 정했는가"를 피처 그룹별 기여도로 설명 (금융 XAI 대응)',
    data_mode=data_mode,
)

explanations = (catalog or {}).get("explanations", {})
if synthetic:
    st.info(
        "현재 SHAP 화면은 evaluation-xai의 **합성 데이터 데모**입니다. "
        "실제 PPO 모델의 SHAP(4개 구간 대표 결정)은 evaluation-xai에서 생성됐고, "
        "산출물이 공유되면 설정(`EVAL_EXPLAIN_ARTIFACT_DIR`)만 바꿔 연결합니다.",
        icon=":material/science:",
    )
if catalog and not explanations:
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
        assets = explanations[f"{run_id}/{decision_id}"]
        target = c3.selectbox("대상 자산", assets, format_func=asset_name)

    exp = load(
        api.explain,
        run_id,
        decision_id,
        target,
        synthetic,
        spinner="SHAP 값 조회 중…",
    )
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
            direction = "올렸습니다" if top[1] >= 0 else "내렸습니다"
            st.info(
                f"가장 큰 영향은 **{top[0]}** 그룹으로, "
                f"비중을 {abs(top[1]):.2%}p {direction}. "
                "학습 배경 대비 모델 출력에 대한 기여이며, 인과관계를 뜻하지는 않습니다.",
                icon=":material/lightbulb:",
            )
            if exp.get("interpretation"):
                st.caption(f"evaluation-xai 해석 원문: {exp['interpretation']}")
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

        plot_key = f"{run_id}/{decision_id}/{target}"
        st.markdown("#### Summary Plot · Force Plot")
        if plot_key in catalog.get("plots", []):
            args = (run_id, decision_id, target)
            left, right = st.columns(2, gap="medium")
            with left, st.container(border=True):
                st.markdown("**Summary Plot** · 피처 그룹별 기여도 크기")
                try:
                    png = api.explain_plot(*args, "summary", synthetic)
                    st.image(png, width="stretch")
                except ApiError as error:
                    st.caption(
                        f"그림을 불러오지 못했습니다: {error.hint or error.message}"
                    )
            with right, st.container(border=True):
                st.markdown("**Force Plot** · 기준 비중에서 예측 비중으로 밀고 당긴 힘")
                try:
                    html = api.explain_plot(*args, "force", synthetic)
                    components.html(html.decode("utf-8"), height=170, scrolling=True)
                except ApiError as error:
                    st.caption(
                        f"그림을 불러오지 못했습니다: {error.hint or error.message}"
                    )
            st.caption(
                "Summary Plot은 선택한 한 결정의 기여도 크기입니다. "
                "전체 기간의 중요도로 일반화하지 않습니다."
            )
        else:
            available = ", ".join(catalog.get("plots", [])) or "없음"
            st.caption(
                "이 결정·자산에는 evaluation-xai가 생성한 그림이 없습니다. "
                f"그림이 있는 항목: {available}"
            )

        with st.expander("규제 대응 메모", icon=":material/policy:"):
            st.markdown(
                "- 모든 추천 비중은 SHAP 기여도로 **결정 근거를 추적**할 수 있습니다.\n"
                "- 기여도의 합 = 예측 비중 − 기준 비중 (가법성). "
                "Safe-Guard 개입 이전의 정책 비중을 설명합니다.\n"
                "- 기여도는 '이 입력이 배경 대비 모델 출력 비중에 어떻게 기여했는가'이며, "
                "인과관계나 매매의 타당성을 증명하지 않습니다."
            )
