"""Plotly 차트 공통 스타일.

색 규칙 (데이터 시각화 가이드):
- 범주(전략·자산군)는 검증된 팔레트를 정해진 순서로 사용하고, 범주가 바뀌어도 같은 대상은 같은 색
- 상태(정상/주의/위험)는 별도 상태색만 쓰고 항상 아이콘·문구와 함께 표시
- 양수/음수(SHAP 기여도)는 파랑↔빨강 발산 팔레트
- 선 2px, 격자는 옅은 실선, 모든 차트는 '표로 보기'를 함께 제공
"""

from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st


@dataclass(frozen=True)
class Palette:
    series: tuple[str, ...]
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    positive: str
    negative: str
    neutral: str


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    positive="#2a78d6",
    negative="#e34948",
    neutral="#c3c2b7",
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    positive="#3987e5",
    negative="#e66767",
    neutral="#383835",
)

# 상태색은 테마와 무관하게 고정
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def palette() -> Palette:
    """현재 사용자가 보고 있는 테마(라이트/다크)에 맞는 팔레트."""
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def color_map(categories: list[str]) -> dict[str, str]:
    """범주 → 색. 순서가 고정된 목록을 받아 같은 범주는 항상 같은 색이 되게 한다."""
    p = palette()
    return {c: p.series[i % len(p.series)] for i, c in enumerate(categories)}


def style(fig: go.Figure, *, height: int = 360, percent_y: bool = False, hover: str = "x unified") -> go.Figure:
    p = palette()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p.ink_secondary, size=13),
        hovermode=hover,
        hoverlabel=dict(bgcolor=p.surface, bordercolor=p.grid, font=dict(color=p.ink)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    # automargin: 축 라벨 길이에 맞춰 여백을 자동 확보 (라벨 잘림 방지)
    fig.update_xaxes(showgrid=False, linecolor=p.axis, tickfont=dict(color=p.muted), zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=p.grid, gridwidth=1, linecolor=p.axis, tickfont=dict(color=p.muted), zeroline=False,
                     automargin=True)
    if percent_y:
        fig.update_yaxes(tickformat=".1%")
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def line(series: dict[str, tuple[list, list]], *, colors: dict[str, str], percent_y: bool = True, height: int = 360):
    """여러 시계열 선 차트. 끝점에만 값 라벨을 달아 색에만 의존하지 않게 한다."""
    p = palette()
    fig = go.Figure()
    for name, (x, y) in series.items():
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(width=2, color=colors[name]),
                                 hovertemplate="%{y:.2%}" if percent_y else "%{y:,.2f}"))
        if x:
            fig.add_annotation(x=x[-1], y=y[-1], text=f"{y[-1]:+.1%}" if percent_y else f"{y[-1]:,.2f}",
                               showarrow=False, xanchor="left", xshift=6, font=dict(color=p.ink, size=12))
    style(fig, height=height, percent_y=percent_y)
    fig.update_layout(margin=dict(r=56))
    return fig


def underwater(x: list, drawdown: list, *, limit: float = 0.15, height: int = 300):
    """낙폭(underwater) 차트. 0 아래로 내려가며, Safe-Guard 한도선을 라벨과 함께 표시."""
    p = palette()
    neg = [-d if d is not None else None for d in drawdown]
    fig = go.Figure(go.Scatter(x=x, y=neg, mode="lines", fill="tozeroy", name="낙폭",
                               line=dict(width=2, color=p.series[0]), fillcolor=_alpha(p.series[0], 0.18),
                               hovertemplate="낙폭 %{y:.2%}<extra></extra>"))
    fig.add_hline(y=-limit, line=dict(color=STATUS["critical"], width=1.5),
                  annotation_text=f"⚠ Safe-Guard 한도 -{limit:.0%}", annotation_position="bottom left",
                  annotation_font=dict(color=STATUS["critical"], size=12))
    style(fig, height=height, percent_y=True)
    lowest = min([v for v in neg if v is not None] + [-limit])
    fig.update_yaxes(range=[lowest * 1.15, 0.005])
    fig.update_layout(showlegend=False)
    return fig


def stacked_area(x: list, groups: dict[str, list], *, colors: dict[str, str], height: int = 320):
    fig = go.Figure()
    for name, y in groups.items():
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", stackgroup="w", line=dict(width=0.5, color=colors[name]),
                                 fillcolor=_alpha(colors[name], 0.85), hovertemplate="%{y:.1%}"))
    style(fig, height=height, percent_y=True)
    fig.update_yaxes(range=[0, 1])
    return fig


def hbar(labels: list[str], values: list[float], *, height: int | None = None, percent: bool = True):
    """정렬된 가로 막대 (단일 계열이라 한 가지 색). 값은 막대 끝 바깥에 표시."""
    p = palette()
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=p.series[0], cornerradius=4),
                           text=[f"{v:.1%}" if percent else f"{v:,.2f}" for v in values], textposition="outside",
                           textfont=dict(color=p.ink), cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2%}<extra></extra>" if percent else "%{y}: %{x:,.3f}<extra></extra>"))
    style(fig, height=height or max(220, 30 * len(labels) + 60), hover="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_layout(bargap=0.35, margin=dict(r=48), showlegend=False)
    return fig


def grouped_bar(x: list[str], groups: dict[str, list[float | None]], *, colors: dict[str, str], percent: bool,
                height: int = 340):
    """구간(x)별로 전략을 나란히 비교. 막대 사이 2px 간격, 끝 모서리 4px."""
    p = palette()
    fig = go.Figure()
    for name, values in groups.items():
        fig.add_trace(go.Bar(x=x, y=values, name=name, marker=dict(color=colors[name], cornerradius=4,
                                                                   line=dict(color=p.surface, width=2)),
                             hovertemplate=f"{name} %{{x}}: " + ("%{y:.2%}" if percent else "%{y:.3f}") + "<extra></extra>"))
    style(fig, height=height, percent_y=percent, hover="closest")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    fig.add_hline(y=0, line=dict(color=p.axis, width=1))
    return fig


def donut(labels: list[str], values: list[float], *, colors: dict[str, str], center: str = "", height: int = 300):
    """구성비 도넛 (6개 이하 범주에서만 사용)."""
    p = palette()
    fills = [colors[label] for label in labels]
    # 노란 계열 위에는 흰 글씨 대비가 부족하므로 검정 글씨
    ink = ["#0b0b0b" if f in ("#eda100", "#c98500") else "#ffffff" for f in fills]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, sort=False, direction="clockwise",
                           marker=dict(colors=fills, line=dict(color=p.surface, width=2)),
                           textinfo="percent", textfont=dict(color=ink), hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.add_annotation(text=center, showarrow=False, font=dict(size=15, color=p.ink))
    style(fig, height=height, hover="closest")
    fig.update_layout(legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left"), margin=dict(t=8))
    return fig


def waterfall(base: float, labels: list[str], contributions: list[float], *, height: int = 360):
    """SHAP 기여도: 기준 비중 → 피처 그룹별 기여 → 예측 비중."""
    p = palette()
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", *["relative"] * len(contributions), "total"],
        x=["기준 비중", *labels, "예측 비중"],
        y=[base, *contributions, 0],
        text=[f"{base:.2%}", *[f"{c:+.2%}" for c in contributions], f"{base + sum(contributions):.2%}"],
        textposition="outside",
        textfont=dict(color=p.ink),
        connector=dict(line=dict(color=p.axis, width=1)),
        increasing=dict(marker=dict(color=p.positive)),
        decreasing=dict(marker=dict(color=p.negative)),
        totals=dict(marker=dict(color=p.muted)),
        hovertemplate="%{x}: %{y:+.3%}<extra></extra>",
        cliponaxis=False,
    ))
    style(fig, height=height, percent_y=True, hover="closest")
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(showlegend=False)
    return fig


def gauge(value: float, *, limit: float, title: str, height: int = 220):
    """한도 대비 현재 수준 게이지. 색은 상태색(정상/주의/위험)으로만 쓰고 제목에 상태 문구를 함께 둔다."""
    p = palette()
    ratio = value / limit if limit else 0
    color = STATUS["good"] if ratio < 0.6 else STATUS["warning"] if ratio < 1 else STATUS["critical"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number=dict(suffix="%", valueformat=".2f", font=dict(color=p.ink, size=30)),
        title=dict(text=title, font=dict(color=p.ink_secondary, size=14)),
        gauge=dict(
            axis=dict(range=[0, max(limit * 100 * 1.25, value * 100 * 1.1)], ticksuffix="%", tickcolor=p.muted, tickfont=dict(color=p.muted)),
            bar=dict(color=color, thickness=0.28),
            bgcolor=p.grid,
            borderwidth=0,
            threshold=dict(line=dict(color=STATUS["critical"], width=3), thickness=0.85, value=limit * 100),
        ),
    ))
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def forest(labels: list[str], means: list[float], lows: list[float], highs: list[float], *, height: int = 260):
    """평균 차이와 95% 신뢰구간. 0선을 기준으로 유의성을 읽는다."""
    p = palette()
    fig = go.Figure(go.Scatter(
        x=means, y=labels, mode="markers", marker=dict(size=10, color=p.series[0], line=dict(color=p.surface, width=2)),
        error_x=dict(type="data", symmetric=False, array=[h - m for h, m in zip(highs, means)],
                     arrayminus=[m - lo for m, lo in zip(means, lows)], color=p.series[0], thickness=2, width=6),
        hovertemplate="%{y}<br>평균 차이 %{x:.3%}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=p.axis, width=1))
    style(fig, height=height, hover="closest")
    fig.update_xaxes(tickformat=".2%", showgrid=True, gridcolor=p.grid)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(showlegend=False)
    return fig


def _alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
