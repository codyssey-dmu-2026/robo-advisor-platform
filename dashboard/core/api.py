"""API Gateway HTTP 클라이언트. 대시보드는 모델을 직접 로드하지 않고 이 모듈로만 데이터를 받는다."""

import os
from typing import Any

import requests
import streamlit as st

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8080").rstrip("/")
# 요청 1건 5초 SLA에 맞춘 타임아웃 (리서치는 LLM 사용 시 오래 걸릴 수 있어 여유를 둠)
TIMEOUT_S = float(os.getenv("DASHBOARD_TIMEOUT_S", "5"))
RESEARCH_TIMEOUT_S = float(os.getenv("DASHBOARD_RESEARCH_TIMEOUT_S", "8"))


class ApiError(Exception):
    """Gateway 호출 실패. 화면에는 message와 hint를 그대로 보여준다."""

    def __init__(
        self, message: str, status: int | None = None, hint: str | None = None
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.hint = hint


def _request(method: str, path: str, *, timeout: float = TIMEOUT_S, **kwargs) -> Any:
    try:
        response = requests.request(
            method, f"{GATEWAY_URL}{path}", timeout=timeout, **kwargs
        )
    except requests.exceptions.Timeout:
        raise ApiError(
            "응답 시간이 초과됐습니다.",
            hint=f"{timeout:.0f}초 안에 응답이 오지 않았습니다.",
        ) from None
    except requests.exceptions.RequestException:
        raise ApiError(
            "API Gateway에 연결할 수 없습니다.",
            hint=f"Gateway({GATEWAY_URL})가 실행 중인지 확인하세요. "
            "docker compose up 후 잠시 기다려 주세요.",
        ) from None
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text[:200]
        raise ApiError(
            f"요청이 실패했습니다 (HTTP {response.status_code}).",
            status=response.status_code,
            hint=str(detail),
        )
    return response.json()


@st.cache_data(ttl=10, show_spinner=False)
def health() -> dict:
    return _request("GET", "/health", timeout=4)


@st.cache_data(ttl=60, show_spinner=False)
def backtest() -> dict:
    return _request("GET", "/backtest")


@st.cache_data(ttl=300, show_spinner=False)
def performance() -> dict:
    """현재 RL 모델의 Walk-Forward 성과 (rl-engine 실제 백테스트 결과)."""
    return _request("GET", "/performance")


@st.cache_data(ttl=60, show_spinner=False)
def timeseries(run_id: str) -> dict:
    return _request("GET", f"/backtest/{run_id}/timeseries")


@st.cache_data(ttl=300, show_spinner=False)
def explain(run_id: str, decision_id: str, target_asset: str) -> dict:
    payload = {
        "run_id": run_id,
        "decision_id": decision_id,
        "target_asset": target_asset,
    }
    return _request("POST", "/explain", json=payload)


@st.cache_data(ttl=30, show_spinner=False)
def optimize(tickers: tuple[str, ...], risk_override: bool, as_of: str | None) -> dict:
    payload = {"tickers": list(tickers), "risk_override": risk_override, "as_of": as_of}
    return _request("POST", "/optimize", json=payload)


@st.cache_data(ttl=300, show_spinner=False)
def risk_tags() -> dict:
    """리스크 태그 Safe-Guard가 현재 모델의 평가 구간에서 작동한 내역."""
    return _request("GET", "/risk-tags")


def research(question: str, tickers: list[str], as_of: str | None) -> dict:
    # 비운 값은 보내지 않아 에이전트가 질문에서 종목을 인식하고 요청 시각을 기준으로 삼게 한다
    payload = {"question": question, "tickers": tickers or None, "as_of": as_of}
    payload = {k: v for k, v in payload.items() if v is not None}
    return _request("POST", "/research", json=payload, timeout=RESEARCH_TIMEOUT_S)
