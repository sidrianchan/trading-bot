from __future__ import annotations

import pandas as pd
import yfinance as yf
from loguru import logger

from data.cache import Cache

_cache = Cache(ttl_hours=72)  # fundamentals change slowly

_FIELDS = {
    "returnOnEquity": "roe",
    "debtToEquity": "debt_equity",
    "trailingPE": "pe_ratio",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",  # YoY EPS growth — proxy for analyst revision
    "marketCap": "market_cap",
    "sector": "sector",
}


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch basic fundamentals for each ticker via yfinance.

    Returns a DataFrame indexed by ticker with columns:
        roe, debt_equity, pe_ratio, revenue_growth, market_cap, sector

    NOTE: yfinance returns point-in-time data for today, not historical.
    For production use, replace with a point-in-time fundamentals provider
    (e.g. Simfin, Tiingo, FactSet) to avoid look-ahead bias in backtests.
    This is acceptable for paper trading and initial strategy validation.
    """
    cache_key = f"fundamentals_{'_'.join(sorted(tickers)[:5])}_n{len(tickers)}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    logger.info(f"Fetching fundamentals for {len(tickers)} tickers")
    rows: list[dict] = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            row = {"ticker": ticker}
            for yf_key, col_name in _FIELDS.items():
                row[col_name] = info.get(yf_key)
            rows.append(row)
        except Exception as exc:
            logger.warning(f"Fundamentals fetch failed for {ticker}: {exc}")
            rows.append({"ticker": ticker})

    df = pd.DataFrame(rows).set_index("ticker")

    # Winsorize at 1st/99th percentile to reduce outlier impact
    for col in ["roe", "debt_equity", "pe_ratio", "revenue_growth"]:
        if col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)

    _cache.set(cache_key, df)
    return df
