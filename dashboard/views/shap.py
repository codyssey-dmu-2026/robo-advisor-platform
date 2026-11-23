"""SHAP 해석: 특정 의사결정에서 어떤 피처 그룹이 자산 비중을 올리고 내렸는가."""

import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core import api, charts
from core.api import ApiError
from core.assets import ASSETS, asset_name
from core.ui import empty_state, load, page_header, table_view

GROUP_LABELS = {
    # 합성 데모의 그룹
    "returns": "과거 수익률",
    "indicators": "기술적 지표 (RSI·MACD)",
    "risk_inputs": "리스크 입력 (낙폭·리스크 태그)",
    # 실제 PPO 관측의 그룹
    "portfolio_weights": "현재 보유 비중 (12개 자산)",
}
FEATURES = {"rsi": "RSI", "macd": "MACD", "is_market_open": "휴장 여부"}
VARIANTS = {"simple": "단순 수익률", "sharpe": "샤프비율"}
TOP = 8  # 폭포 차트에 따로 그릴 그룹 수 (나머지는 합쳐서 한 막대)


def short_name(ticker: str) -> str:
    """차트용 짧은 이름: 미국 자산은 티커, 한국 자산은 종목명."""
    asset = ASSETS.get(ticker)
    return asset.name if asset and asset.region == "한국" else ticker


def group_label(group: str, short: bool = False) -> str:
    """'AAPL:returns' → 'Apple (AAPL) 과거 20일 수익률' (short면 'AAPL 수익률')."""
    if group == "portfolio_weights" and short:
        return "보유 비중"
    if group in GROUP_LABELS:
        return GROUP_LABELS[group]
    name = short_name if short else asset_name
    ticker, _, kind = group.partition(":")
    if kind == "returns":
        return f"{name(ticker)} {'수익률' if short else '과거 20일 수익률'}"
    if ticker == "static":
        asset, _, feature = kind.partition("_")
        return f"{name(asset)} {FEATURES.get(feature, feature)}"
    return group


def run_label(run_id: str) -> str:
    """'ppo_test_w4_mdd_l1p0_s42' → '2025년 테스트 모델 · MDD 페널티 λ=1.0 · seed 42'."""
    found = re.fullmatch(r"ppo_test_w(\d+)_(.+)_s(\d+)", run_id)
    if not found:
        return run_id
    window, variant, seed = found.groups()
    if variant.startswith("mdd_l"):
        variant = f"MDD 페널티 λ={variant[5:].replace('p', '.')}"
    variant = VARIANTS.get(variant, variant).replace("MDD 페널티", "MDD")
    return f"{2021 + int(window)}년 · {variant} · seed {seed}"


def decision_label(decision_id: str) -> str:
    found = re.fullmatch(r"d(\d{4})(\d{2})(\d{2})", decision_id)
    return "-".join(found.groups()) if found else decision_id


catalog = load(api.explain_catalog)
data_mode = catalog.get("data_mode") if catalog else None
# 합성 데모 설명은 evaluation-xai가 명시적으로 허용해야만 돌려준다 (실제 의사결정 설명과 구분)
synthetic = data_mode == "synthetic"
provisional = str((catalog or {}).get("status") or "").startswith("provisional")
page_header(
    "SHAP 의사결정 해석",
    '"왜 이 비중을 정했는가"를 피처 그룹별 기여도로 설명 (금융 XAI 대응)',
    # 실제 결과지만 최종 검증 전이면 '잠정'으로 표시
    data_mode=(
        "imported_unverified" if data_mode == "real" and provisional else data_mode
    ),
)

explanations = (catalog or {}).get("explanations", {})
if catalog and not synthetic and explanations:
    st.caption(
        ":material/verified: 학습된 PPO 모델의 실제 결정을 evaluation-xai가 설명한 결과입니다 "
        f"(결정 {len(explanations)}건 × 12개 자산, 그림은 SPY). 설명 대상 모델은 "
        "MDD 페널티(λ=1.0) 보상 모델로, 현재 서빙 모델(단순 수익률)과는 다른 모델입니다."
    )
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
        run_id = c1.selectbox("실행 (run)", run_ids, format_func=run_label)
        decisions = [k.split("/")[1] for k in keys if k.startswith(f"{run_id}/")]
        decision_id = c2.selectbox(
            "의사결정 시점 (decision)", decisions, format_func=decision_label
        )
        assets = explanations[f"{run_id}/{decision_id}"]
        # 그림이 있는 자산을 먼저 보여 준다
        drawn = [a for a in assets if f"{run_id}/{decision_id}/{a}" in catalog["plots"]]
        target = c3.selectbox(
            "대상 자산",
            assets,
            index=assets.index(drawn[0]) if drawn else 0,
            format_func=asset_name,
        )

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
            help="비교 기준: 학습 구간 배경 관측에서 이 자산의 평균 비중 "
            "(직전 보유 비중이 아님)",
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
            labels = [group_label(g) for g in groups]
            order = sorted(range(len(groups)), key=lambda i: -abs(contributions[i]))
            ranked = [(labels[i], contributions[i]) for i in order]
            # 차트에는 짧은 이름, 표에는 전체 이름
            shown = [
                (group_label(groups[i], short=True), contributions[i])
                for i in order[:TOP]
            ]
            if len(ranked) > TOP:
                rest = ranked[TOP:]
                shown.append((f"나머지 {len(rest)}개", sum(v for _, v in rest)))
            with st.container(border=True):
                st.markdown(f"**{asset_name(target)} 비중 결정 요인**")
                caption = "파란 막대는 비중을 올린 요인, 빨간 막대는 내린 요인입니다."
                if len(ranked) > TOP:
                    share = sum(abs(v) for _, v in ranked[:TOP]) / max(
                        sum(abs(v) for _, v in ranked), 1e-12
                    )
                    caption += (
                        f" 영향이 큰 {TOP}개 그룹(전체 영향의 {share:.0%})만 따로 그리고 "
                        "나머지는 합쳤습니다."
                    )
                st.caption(caption)
                charts.show(
                    charts.waterfall(
                        base, [n for n, _ in shown], [v for _, v in shown]
                    ),
                    key="shap_wf",
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
                pd.DataFrame(ranked, columns=["피처 그룹", "SHAP 기여도"]),
                label=f"표로 보기 (그룹 {len(ranked)}개 전체, 영향 큰 순)",
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
