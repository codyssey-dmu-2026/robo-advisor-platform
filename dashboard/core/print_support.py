"""브라우저 인쇄(⋮ → Print, Ctrl+P) 대응.

차트(Plotly)는 화면에서 그려진 픽셀 너비를 그대로 유지하므로, 더 좁은 용지에서는 오른쪽이 잘린다.
본문을 축소(zoom)하면 페이지 나눔에서 차트와 카드가 어긋나므로, 대신 인쇄 직전에 용지 크기를
본문 너비에 맞춘다 (A4 가로 비율 유지). PDF 저장 시 잘림이 없고, 실제 프린터는 용지에 맞춰 축소한다.
사이드바·툴바·접힌 표는 인쇄에서 숨긴다.
"""

import streamlit as st

_PRINT_CSS = """
<style>
@media print {
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
  [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stExpander"], [data-testid="stStatusWidget"] {
    display: none !important;
  }
  [data-testid="stMain"] { margin-left: 0 !important; }
  [data-testid="stMainBlockContainer"] { padding: 0 !important; }
  /* 세로 flex 배치는 인쇄 쪽 나눔에서 항목만 다음 쪽으로 밀려 카드와 겹치므로 블록 배치로 바꿈 */
  [data-testid="stVerticalBlock"] { display: block !important; }
  [data-testid="stVerticalBlock"] > * { margin-bottom: 1rem; }
  /* 차트·KPI가 페이지 경계에서 둘로 잘리지 않게 */
  [data-testid="stPlotlyChart"], [data-testid="stMetric"] { break-inside: avoid; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
</style>
"""

_PRINT_JS = """
<script>
(() => {
  if (window.__roboPrintFit) return;  // 페이지가 다시 그려져도 한 번만 등록
  window.__roboPrintFit = true;
  const MARGIN_PX = 38;  // 약 10mm
  window.addEventListener("beforeprint", () => {
    const el = document.querySelector('[data-testid="stMainBlockContainer"]');
    if (!el) return;
    // 화면 기준 본문 너비(차트 픽셀 너비 포함)에 맞춰 용지 크기 지정, 최소 A4 가로
    const width = Math.max(el.scrollWidth + MARGIN_PX * 2, 1123);
    const height = Math.round(width * 210 / 297);
    let style = document.getElementById("robo-print-page");
    if (!style) {
      style = document.createElement("style");
      style.id = "robo-print-page";
      document.head.appendChild(style);
    }
    style.textContent =
      `@page { size: ${width}px ${height}px; margin: ${MARGIN_PX}px; }`;
  });
})();
</script>
"""


def enable_print_layout() -> None:
    st.html(_PRINT_CSS)
    # 직접 작성한 고정 스크립트만 실행 (사용자 입력 없음)
    st.html(_PRINT_JS, unsafe_allow_javascript=True)
