"""대시보드·시계열 API 테스트.

- 시계열 API: 임시 산출물 폴더로 정상/없는 실행/잘못된 ID를 검증
- 대시보드: Gateway가 꺼져 있어도 모든 페이지가 예외 없이 안내 화면을 보여주는지 검증
"""

import os
from pathlib import Path

# 대시보드 모듈이 처음 import될 때 주소를 읽으므로, 연결되지 않는 주소를 먼저 지정
os.environ["GATEWAY_URL"] = "http://127.0.0.1:9"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from conftest import risk_tag  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import gateway  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
PAGES = [
    "views/portfolio.py",
    "views/risk.py",
    "views/performance.py",
    "views/shap.py",
    "views/anova.py",
    "views/research.py",
]


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, freq="B", name="date")
    run = tmp_path / "mvo_WF2024"
    run.mkdir()
    pd.DataFrame(
        {
            "equity_start": [100.0, 101.0, 99.0],
            "equity_end": [101.0, 99.0, 100.0],
            "net_return_simple": [0.01, -0.0198, 0.0101],
            "drawdown": [0.0, 0.0198, 0.0099],
            "safeguard_triggered": [False, False, False],
            "guard_active": [False, False, False],
        },
        index=dates,
    ).to_parquet(run / "ledger.parquet")
    pd.DataFrame(
        {"SPY": [0.6, 0.5, 0.5], "CASH": [0.4, 0.5, 0.5]}, index=dates
    ).to_parquet(run / "weights.parquet")
    monkeypatch.setattr(gateway, "EVAL_ARTIFACT_DIR", tmp_path)
    return tmp_path


def test_15_timeseries_returns_aligned_series(artifact_dir):
    """15. 시계열 API가 날짜·누적수익률·낙폭·비중을 같은 길이로 반환하는지 검증"""
    data = TestClient(gateway.app).get("/backtest/mvo_WF2024/timeseries").json()
    assert data["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert data["cumulative_return"][0] == pytest.approx(0.01)
    assert len(data["drawdown"]) == len(data["dates"]) == len(data["weights"]["SPY"])
    assert "투자 조언" in data["disclaimer"]


def test_16_timeseries_unknown_and_invalid_run(artifact_dir):
    """16. 없는 실행은 404, 경로 조작이 가능한 ID는 422로 거부하는지 검증"""
    client = TestClient(gateway.app)
    assert client.get("/backtest/nope/timeseries").status_code == 404
    assert client.get("/backtest/bad.id/timeseries").status_code == 422


class _Sent(list):
    """보낸 요청 기록 + 가짜 Gateway 응답 표."""

    responses: dict


class _Response:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        # 그림(PNG·HTML) 응답은 바이트로 받는다
        self.content = payload if isinstance(payload, bytes) else b""

    def json(self) -> dict:
        return self._payload


# evaluation-xai import-rl 산출물과 같은 구조 (값은 실제 분석 결과)
IMPORTED_BACKTEST = {
    "data_mode": "imported_unverified",
    "status": "provisional",
    "runs": {"EqualWeight_w1": {"strategy": "EqualWeight", "metrics": {}}},
    "explanations": {},
    "mission_anova": {
        "reward_variants": {
            "status": "complete",
            "effects": [
                {
                    "term": "C(strategy)",
                    "f": 0.19656,
                    "p_value": 0.82178,
                    "eta_squared": 0.00278,
                }
            ],
            "diagnostics": {"levene_p": 0.6056, "shapiro_residual_p": 0.0741},
            "tukey_status": "not_required",
            "representative_mdd_lambda": 1.0,
        },
        "strategies": {"status": "incomplete", "reason": "MVO missing"},
        "strategy_by_regime": {"status": "incomplete", "reason": "MVO missing"},
    },
    "bootstrap": [
        {
            "mean_monthly_difference": -0.00365,
            "ci_95": [-0.00519, -0.00188],
            "block_length": n,
            "repetitions": 5000,
            "comparison": "PPO-mdd_l1.0_seed_mean_minus_EqualWeight_daily",
        }
        for n in (1, 3, 6)
    ],
    "limitations": ["Imported producer-declared real results."],
}
SHAP_EXPLAIN = {
    "data_mode": "synthetic",
    "run_id": "toy_policy",
    "decision_id": "toy0",
    "target_asset": "SYN00",
    "groups": ["returns", "indicators", "risk_inputs"],
    "base_value": 0.1,
    "predicted_weight": 0.15,
    "contributions": [0.03, 0.01, 0.01],
}
PLOTS = "/explain/plots/toy_policy/toy0/SYN00"


def _learning_curves() -> dict:
    """/learning-curves 응답 형식: 현재 모델 구간의 단순 수익률 seed 2개."""
    import rl_server

    window = int(rl_server.RL_MODEL_NAME.split("_w")[1].split("_")[0])
    rows = [
        {
            "run": f"ppo_test_w{window}_simple_s{seed}",
            "stage": "test",
            "window": window,
            "variant": "simple",
            "seed": seed,
            "attempt": "PPO_1",
            "step": step,
            "ep_rew_mean": step / 1e6,
            "ep_len_mean": 240.0,
        }
        for seed in (42, 43)
        for step in range(20480, 102400 + 1, 20480)
    ]
    return rl_server.build_learning_curves(pd.DataFrame(rows))


def _guard_backtest() -> dict:
    """/risk-tags/backtest 응답 형식 (켬 쪽 누적 +2%p, MDD −1%p)."""
    keys = [
        "cumulative_return",
        "cagr",
        "annualized_volatility",
        "var_95",
        "cvar_95",
        "mdd",
        "sharpe",
        "sortino",
        "calmar",
        "avg_turnover",
    ]
    off = dict.fromkeys(keys, 0.1) | {"cumulative_return": 0.40, "mdd": 0.17}
    on = off | {"cumulative_return": 0.42, "mdd": 0.16, "sharpe": 0.12}
    dates = ["2025-01-03", "2025-01-06"]
    return {
        "status": "ok",
        "model_name": "fake",
        "period": {"start": dates[0], "end": dates[-1]},
        "trading_days": 2,
        "method": "폐루프 재생",
        "adjusted_days": 1,
        "metrics": {"model": off, "guarded": on},
        "difference": {k: on[k] - off[k] for k in keys},
        "series": {
            "dates": dates,
            "model": {"cumulative_return": [0.0, 0.40], "drawdown": [0.0, 0.0]},
            "guarded": {"cumulative_return": [0.0, 0.42], "drawdown": [0.0, 0.0]},
        },
        "reproduces_published": {"matched_days": 2, "max_abs_diff": 2e-7},
        "limitation": "평가 구간 1년만 비교한 결과입니다.",
        "disclaimer": "교육 목적",
    }


def _png() -> bytes:
    """1×1 PNG (st.image가 읽을 수 있는 최소 이미지)."""
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGA"
        "WjR9awAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def online_gateway(perf_results, write_tags, monkeypatch):
    """Gateway 응답을 실제 형식의 성과 데이터로 대체 (RL 성과 화면 렌더링 검증용)."""
    import requests
    import streamlit as st

    import rl_server

    monkeypatch.setattr(
        rl_server, "get_schedule", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    perf = rl_server.get_performance() | {"weights": None}
    tickers = [
        "000660",
        "005930",
        "069500",
        "148070",
        "AAPL",
        "GLD",
        "MSFT",
        "NVDA",
        "QQQ",
        "SPY",
        "TLT",
        "VNQ",
    ]
    # 리스크 태그 Safe-Guard: 01-03 장 마감에 NVDA 위험 신호(점수 0.8)가 공개된 상황
    schedule = rl_server.PolicySchedule(
        model_name="fake",
        tickers=tickers,
        dates=pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"]),
        weights=np.full((3, len(tickers)), 1 / len(tickers)),
    )
    write_tags(risk_tag("NVDA", 0.8, "2025-01-03T16:00:00-05:00"))
    tags = rl_server.get_risk_tags()
    decision = rl_server.guard_decision(schedule, 2, tags)
    responses = {
        "/performance": perf,
        "/backtest": IMPORTED_BACKTEST,
        "/explain/catalog": {
            "data_mode": "synthetic",
            "explanations": {"toy_policy/toy0": ["SYN00"]},
            "plots": ["toy_policy/toy0/SYN00"],
            "disclaimer": "교육 목적",
        },
        "/explain": SHAP_EXPLAIN,
        f"{PLOTS}/summary": _png(),
        f"{PLOTS}/force": b"<html><body>force plot</body></html>",
        "/risk-tags": rl_server.build_guard_summary(schedule, tags),
        "/risk-tags/backtest": _guard_backtest(),
        "/learning-curves": _learning_curves(),
        "/health": {
            "gateway": "ok",
            "services": {
                "rl_engine": {
                    "status": "ok",
                    "as_of_min": "2025-01-03",
                    "as_of_max": "2025-01-07",
                }
            },
        },
        "/optimize": {
            "status": "ok",
            "weights": dict(zip(tickers, decision["weights"].tolist())),
            "model_weights": dict(zip(tickers, schedule.weights[2].tolist())),
            "decision_date": "2025-01-06",
            "model_used": "fake",
            "risk_override_applied": True,
            "risk_guard": {
                "requested": True,
                "available": True,
                "rule": rl_server._guard_rule(schedule),
                "moved_weight": decision["moved_weight"],
                "adjustments": decision["adjustments"],
                "tags": decision["tags"],
            },
        },
    }

    def fake_request(method, url, **kwargs):
        path = url.removeprefix(os.environ["GATEWAY_URL"])
        if path not in responses:
            raise requests.exceptions.ConnectionError(path)
        sent.append((path, kwargs))
        return _Response(responses[path])

    sent = _Sent()
    sent.responses = responses  # 테스트에서 응답을 바꿔 끼울 수 있게

    monkeypatch.setattr(requests, "request", fake_request)
    st.cache_data.clear()
    yield sent
    st.cache_data.clear()


@pytest.mark.parametrize(
    "page", [None, "views/portfolio.py", "views/risk.py", "views/performance.py"]
)
def test_18_dashboard_renders_real_performance(online_gateway, page):
    """18. 실제 성과 데이터로 개요·포트폴리오·리스크·강화학습 성과 화면이 예외 없이 그려지는지 검증"""
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    texts = [m.value for m in at.markdown]
    # st.badge는 markdown으로 그려진다. 실데이터 표시가 있고 API 오류 카드는 없어야 함
    assert any("실데이터" in t for t in texts)
    failures = ("연결할 수 없습니다", "요청이 실패했습니다", "응답 시간이 초과")
    assert not any(f in t for t in texts for f in failures), "API 오류 카드가 표시됨"


def _page(path: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.switch_page(path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_19_portfolio_shows_risk_tag_adjustments(online_gateway):
    """19. 포트폴리오 화면이 리스크 태그 Safe-Guard의 조정 내역과 발동일 바로가기를 보여 주는지 검증"""
    at = _page("views/portfolio.py")
    texts = [m.value for m in at.markdown]
    assert any("리스크 태그 Safe-Guard 조정 내역" in t for t in texts)
    guard_kpi = next(m for m in at.metric if m.label == "리스크 태그 Safe-Guard")
    assert guard_kpi.value == "1개 축소"
    table = next(d.value for d in at.dataframe if "조정 후" in d.value.columns)
    assert table["조정 후"].iloc[0] == pytest.approx(1 / 12 * (1 - 0.5 * 0.8))
    assert at.toggle[0].value is True  # 기본값: 리스크 태그 반영 켜짐

    # 발동일 바로가기를 고르면 기준일이 그 날짜로 바뀐다
    jump = at.selectbox(key="pf_jump")
    assert list(jump.options) and jump.value is None
    jump.select("2025-01-04").run()
    assert not at.exception, [e.value for e in at.exception]
    assert str(at.date_input(key="pf_as_of").value) == "2025-01-04"


def test_20_risk_page_shows_risk_tag_guard_summary(online_gateway):
    """20. 리스크 화면이 리스크 태그 Safe-Guard 발동 현황(발동일 수)과 한계 안내를 보여 주는지 검증"""
    at = _page("views/risk.py")
    days = next(m for m in at.metric if m.label == "발동일")
    assert days.value == "2일"
    assert any("백테스트 성과에는" in c.value for c in at.caption)


def test_24_performance_page_shows_learning_curves(online_gateway):
    """24. 강화학습 성과 화면의 학습 곡선 탭이 현재 모델 요약과 보상 함수별 요약 표를 보여 주는지 검증"""
    at = _page("views/performance.py")
    assert any("현재 서빙 모델" in c.value for c in at.caption)
    table = next(d.value for d in at.dataframe if "마지막 평균 보상" in d.value.columns)
    assert len(table) == 2 and set(table["seed"]) == {42, 43}
    assert table["학습 스텝"].min() >= 100_000
    assert any("기록 없음" in c.value for c in at.caption)  # 다른 보상 함수 칸


def test_23_risk_page_compares_guard_on_and_off(online_gateway):
    """23. 리스크 화면이 Safe-Guard 끔/켬 백테스트의 지표 차이와 공개 결과 대조를 보여 주는지 검증"""
    at = _page("views/risk.py")
    cum = next(m for m in at.metric if m.label == "누적 수익률 (켬)")
    assert cum.value == "42.0%" and "+2.0%p" in cum.delta
    mdd = next(m for m in at.metric if m.label == "최대 낙폭 MDD (켬)")
    assert "-1.0%p" in mdd.delta
    assert any("2거래일 대조" in c.value for c in at.caption)
    table = next(d.value for d in at.dataframe if "차이 (켬 − 끔)" in d.value.columns)
    assert len(table) == 10


def test_21_anova_page_shows_real_reward_test(online_gateway):
    """21. ANOVA 화면이 실제 분석의 검증 1 수치·비유의 해석·부트스트랩을 보여 주고 나머지는 대기로 두는지 검증"""
    at = _page("views/anova.py")
    texts = [m.value for m in at.markdown]
    assert any("실데이터 · 잠정" in t for t in texts)
    assert any("F = 0.197" in c.value for c in at.caption)
    assert any("MVO 결과가 필요" in c.value for c in at.caption)
    assert any("우연 범위를 벗어나지 않습니다" in i.value for i in at.info)
    effects = next(d.value for d in at.dataframe if "η²" in d.value.columns)
    assert effects["요인"].iloc[0] == "보상 함수"
    assert effects["크기"].iloc[0] == "거의 없음"
    assert any("뚜렷하게 낮습니다" in c.value for c in at.caption)


def test_22_shap_page_requests_demo_with_opt_in_and_shows_plots(online_gateway):
    """22. SHAP 화면이 데모 설명을 명시적으로 허용해 요청하고 Summary·Force Plot을 그리는지 검증"""
    at = _page("views/shap.py")
    texts = [m.value for m in at.markdown]
    assert any("DEMO" in t for t in texts)
    assert any("합성 데이터 데모" in i.value for i in at.info)
    explain = next(kw for path, kw in online_gateway if path == "/explain")
    assert explain["json"]["allow_synthetic"] is True
    plots = [kw for path, kw in online_gateway if path.startswith(PLOTS)]
    assert len(plots) == 2
    assert all(kw["params"]["allow_synthetic"] == "true" for kw in plots)
    assert any("Summary Plot" in t for t in texts)
    assert at.metric[1].value == "15.00%"


def _effect(term, f, p, eta, holm):
    return {
        "term": term,
        "f": f,
        "p_value": p,
        "eta_squared": eta,
        "p_holm_family": holm,
    }


# evaluation-xai compare-rl 결과와 같은 구조 (값은 실제 결과)
COMPARE_BACKTEST = {
    "data_mode": "real",
    "status": "provisional_reference_mismatch",
    "runs": {"MVO_monthly_w1": {"strategy": "MVO", "metrics": {}}},
    "mission_anova": {
        "reward_variants": {
            "status": "complete",
            "effects": [_effect("C(strategy)", 0.19656, 0.82178, 0.00278, 1.0)],
            "diagnostics": {"levene_p": 0.61, "shapiro_residual_p": 0.07},
            "tukey_status": "not_required",
            "tukey": [],
            "representative_mdd_lambda": 1.0,
            "input_reconciliation": {
                "status": "includes_reference_mismatch",
                "mismatched_runs": ["ppo_test_w3_simple_s44"],
                "run_count": 36,
            },
        },
        "strategies": {
            "status": "complete",
            "effects": [_effect("C(strategy)", 0.54528, 0.58090, 0.00768, 1.0)],
            "diagnostics": {"levene_p": 0.0005, "shapiro_residual_p": 0.20},
            "tukey_status": "not_required",
            "tukey": [],
        },
        "strategy_by_regime": {
            "status": "complete",
            "effects": [
                _effect("C(strategy, Sum)", 0.83364, 0.43669, 0.01115, 1.0),
                _effect("C(regime, Sum)", 6.15266, 0.00277, 0.08232, 0.01386),
                _effect("C(strategy, Sum):C(regime, Sum)", 0.2585, 0.904, 0.0069, 1.0),
            ],
            "diagnostics": {"levene_p": 0.0069, "shapiro_residual_p": 0.41},
            "tukey_status": "performed",
            "tukey": [
                {
                    "group1": "EqualWeight / down",
                    "group2": "EqualWeight / up",
                    "mean_difference": -0.03,
                    "p_adjusted": 0.8,
                    "lower": -0.09,
                    "upper": 0.03,
                    "reject": False,
                }
            ]
            * 36,
        },
        "holm_status": "complete",
        "holm_family_size": 5,
        "regime_month_counts": {"up": 23, "sideways": 18, "down": 7},
    },
    "bootstrap": [
        {
            "mean_monthly_difference": mean,
            "ci_95": ci,
            "block_length": 3,
            "repetitions": 5000,
            "comparison": f"PPO-mdd_l1.0_seed_mean_minus_{name}_monthly",
        }
        for name, mean, ci in [
            ("MVO", 0.0062, [-0.0015, 0.0137]),
            ("EqualWeight", -0.0036, [-0.0054, -0.0016]),
        ]
    ],
    "limitations": ["FX/prices/RF/timing remain unverified."],
}
REAL_GROUPS = ["GLD:returns", "000660:returns", "static:MSFT_rsi", "portfolio_weights"]
REAL_GROUPS += [f"static:X{i}_macd" for i in range(10)]


def test_25_anova_page_shows_all_three_tests_with_regime_effect(online_gateway):
    """25. ANOVA 화면이 검증 1·2·3을 모두 완료로 보여 주고, 검증 3의 유의한 국면 효과·Holm·사후 검정을 드러내는지 검증"""
    online_gateway.responses["/backtest"] = COMPARE_BACKTEST
    at = _page("views/anova.py")
    texts = [m.value for m in at.markdown]
    assert any("실데이터 · 잠정" in t for t in texts)
    assert sum("유의한 효과 있음" in t for t in texts) == 1  # 검증 3 카드
    captions = [c.value for c in at.caption]
    assert any(c.startswith("✅ 시장 국면: F = 6.153") for c in captions)
    assert not any("결과 대기" in t for t in texts)
    tables = [d.value for d in at.dataframe if "요인" in d.value.columns]
    assert list(tables[2]["요인"]) == ["전략", "시장 국면", "전략 × 국면"]
    assert (
        tables[1]["요인"].iloc[0] == "전략"
    )  # 검증 2의 C(strategy)는 보상 함수가 아님
    assert "Holm p" in tables[2].columns
    infos = [i.value for i in at.info]
    assert any(
        "시장 국면 효과가 유의합니다" in i and "유의한 쌍은 0개" in i for i in infos
    )
    assert any("36쌍" in c for c in captions)
    assert any("등분산 가정이 기각" in c for c in captions)
    assert any("ppo_test_w3_simple_s44" in c for c in captions)
    assert any(
        "PPO(MDD λ=1.0) − MVO" in c and "뚜렷하지 않습니다" in c for c in captions
    )
    assert any("월간 동일가중" in c and "뚜렷하게 낮습니다" in c for c in captions)
    assert any("잠정" in c for c in captions)


def test_26_shap_page_shows_real_explanation_without_opt_in(online_gateway):
    """26. 실제 SHAP 설명이면 데모 허용 없이 요청하고, 그룹 이름을 한국어로·상위 8개만 따로 그리는지 검증"""
    run, decision = "ppo_test_w4_mdd_l1p0_s42", "d20250616"
    contributions = [0.023, -0.020, 0.005, 0.004] + [0.0001] * 10
    online_gateway.responses.update(
        {
            "/explain/catalog": {
                "data_mode": "real",
                "status": "provisional_reference_mismatch",
                "explanations": {f"{run}/{decision}": ["AAPL", "SPY"]},
                "plots": [f"{run}/{decision}/SPY"],
                "disclaimer": "교육 목적",
            },
            "/explain": {
                "data_mode": "real",
                "run_id": run,
                "decision_id": decision,
                "target_asset": "SPY",
                "groups": REAL_GROUPS,
                "base_value": 0.0679,
                "predicted_weight": 0.0679 + sum(contributions),
                "contributions": contributions,
            },
            f"/explain/plots/{run}/{decision}/SPY/summary": _png(),
            f"/explain/plots/{run}/{decision}/SPY/force": b"<html>force</html>",
        }
    )
    at = _page("views/shap.py")
    explain = next(kw for path, kw in online_gateway if path == "/explain")
    assert explain["json"]["allow_synthetic"] is False
    assert explain["json"]["target_asset"] == "SPY"  # 그림이 있는 자산을 기본 선택
    assert not at.info or not any("합성 데이터 데모" in i.value for i in at.info)
    assert any(
        "현재 서빙 모델(단순 수익률)과는 다른 모델" in c.value for c in at.caption
    )
    assert at.selectbox[0].format_func(run) == "2025년 · MDD λ=1.0 · seed 42"
    assert at.selectbox[1].format_func(decision) == "2025-06-16"
    table = next(d.value for d in at.dataframe if "피처 그룹" in d.value.columns)
    assert table["피처 그룹"].iloc[0] == "금 ETF (GLD) 과거 20일 수익률"
    assert "현재 보유 비중 (12개 자산)" in set(table["피처 그룹"])
    assert any("영향이 큰 8개 그룹" in c.value for c in at.caption)


@pytest.mark.parametrize("page", [None, *PAGES])
def test_17_dashboard_pages_render_when_gateway_offline(page):
    """17. Gateway 오프라인에서도 각 페이지가 예외 없이 렌더링되고 사이드바에 오프라인 안내가 뜨는지 검증"""
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.error, "사이드바에 Gateway 오프라인 안내가 없음"
