"""자산 유니버스 메타데이터 (data-pipeline README의 12종목 기준)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str
    region: str


# 자산군은 5개로 묶어 차트 색 개수를 8개 이하로 유지 (12개 종목을 각각 색칠하지 않음)
ASSET_CLASSES = ["미국 주식", "한국 주식", "채권", "원자재", "리츠"]

ASSETS = {
    a.ticker: a
    for a in [
        Asset("SPY", "S&P500 ETF", "미국 주식", "미국"),
        Asset("QQQ", "나스닥100 ETF", "미국 주식", "미국"),
        Asset("AAPL", "Apple", "미국 주식", "미국"),
        Asset("MSFT", "Microsoft", "미국 주식", "미국"),
        Asset("NVDA", "NVIDIA", "미국 주식", "미국"),
        Asset("TLT", "미국 장기국채 ETF", "채권", "미국"),
        Asset("GLD", "금 ETF", "원자재", "미국"),
        Asset("VNQ", "리츠 ETF", "리츠", "미국"),
        Asset("005930", "삼성전자", "한국 주식", "한국"),
        Asset("000660", "SK하이닉스", "한국 주식", "한국"),
        Asset("069500", "KODEX 200", "한국 주식", "한국"),
        Asset("148070", "KODEX 국고채10년", "채권", "한국"),
    ]
}

DEFAULT_TICKERS = sorted(ASSETS)


def asset_name(ticker: str) -> str:
    asset = ASSETS.get(ticker)
    return f"{asset.name} ({ticker})" if asset else ticker


def asset_class(ticker: str) -> str:
    if ticker == "CASH":
        return "현금"
    asset = ASSETS.get(ticker)
    # evaluation-xai synthetic 데모 자산(SYN00 등)은 자산군 정보가 없음
    return asset.asset_class if asset else "데모 자산"
