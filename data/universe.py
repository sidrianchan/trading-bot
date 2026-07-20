from __future__ import annotations

import io

import pandas as pd
import requests
from loguru import logger

from data.cache import Cache

_cache = Cache(ttl_hours=24)
_CACHE_KEY = "sp500_tickers"
_R1000_CACHE_KEY = "russell1000_tickers"

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)"}

# High-beta names added to the intraday scan regardless of index membership
_HIGH_BETA_ADDITIONS: list[str] = [
    "TSLA", "NVDA", "AMD", "MSTR", "COIN", "SMCI", "PLTR", "MARA", "RIOT",
    "HOOD", "SOFI", "UPST", "AFRM", "RKLB",
]


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    cached = _cache.get(_CACHE_KEY)
    if cached is not None:
        return cached["ticker"].tolist()

    logger.info("Fetching S&P 500 constituents from Wikipedia")
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()

    _cache.set(_CACHE_KEY, pd.DataFrame({"ticker": tickers}))
    logger.info(f"Fetched {len(tickers)} S&P 500 constituents")
    return tickers


def get_russell1000_tickers() -> list[str]:
    """Fetch current Russell 1000 constituents from iShares IWB holdings CSV."""
    cached = _cache.get(_R1000_CACHE_KEY)
    if cached is not None:
        return cached["ticker"].tolist()

    logger.info("Fetching Russell 1000 constituents from iShares IWB")
    try:
        # iShares IWB ETF holdings — publicly available CSV download
        url = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()

        # iShares CSV has metadata rows at top; find the header row
        lines = resp.text.splitlines()
        header_idx = next(i for i, l in enumerate(lines) if "Ticker" in l)
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
        tickers = (
            df["Ticker"]
            .dropna()
            .str.strip()
            .str.replace(".", "-", regex=False)
            .loc[lambda s: s.str.match(r"^[A-Z\-]{1,6}$")]
            .unique()
            .tolist()
        )
    except Exception as exc:
        logger.warning(f"iShares IWB fetch failed ({exc}); falling back to S&P 500 only")
        return get_sp500_tickers()

    _cache.set(_R1000_CACHE_KEY, pd.DataFrame({"ticker": tickers}))
    logger.info(f"Fetched {len(tickers)} Russell 1000 constituents")
    return tickers


def get_intraday_universe(sources: list[str] | None = None) -> list[str]:
    """Combine S&P 500, Russell 1000, and high-beta additions, deduplicated."""
    sources = sources or ["sp500", "russell1000"]
    tickers: set[str] = set(_HIGH_BETA_ADDITIONS)
    if "sp500" in sources:
        tickers.update(get_sp500_tickers())
    if "russell1000" in sources:
        tickers.update(get_russell1000_tickers())
    return sorted(tickers)


def apply_size_filter(
    bars_by_ticker: dict,
    min_median_dollar_vol: float = 10_000_000.0,
    window: int = 60,
    min_price: float = 5.0,
    min_bars: int = 250,
) -> dict:
    """Drop small/illiquid names using a POINT-IN-TIME size proxy.

    Size is proxied by rolling median dollar volume computed from the daily bars
    themselves, so the filter uses only information available on each date.
    Deliberately NOT yfinance `.info` market cap: that is today's value applied
    retroactively to historical dates, which injects look-ahead bias straight
    into the universe definition.

    A ticker is kept if the median of its rolling `window`-day median dollar
    volume clears the floor over the sample. Returns the filtered dict.

    NOTE: the candidate pool itself (get_sp500_tickers / get_russell1000_tickers)
    is *current* index membership, so the universe remains survivorship-biased
    upward. This filter does not fix that, and studies built on it must be framed
    as differences between cohorts rather than absolute performance claims.
    """
    kept: dict = {}
    dropped_short = dropped_cheap = dropped_illiquid = 0

    for ticker, df in bars_by_ticker.items():
        if df is None or len(df) < min_bars:
            dropped_short += 1
            continue
        if float(df["close"].median()) < min_price:
            dropped_cheap += 1
            continue
        dollar_vol = df["close"] * df["volume"]
        rolling = dollar_vol.rolling(window, min_periods=max(20, window // 3)).median()
        if float(rolling.median()) < min_median_dollar_vol:
            dropped_illiquid += 1
            continue
        kept[ticker] = df

    logger.info(
        f"Size filter: {len(bars_by_ticker)} → {len(kept)} tickers "
        f"(dropped {dropped_short} short-history, {dropped_cheap} sub-${min_price:.0f}, "
        f"{dropped_illiquid} below ${min_median_dollar_vol/1e6:.0f}M median $vol)"
    )
    return kept


def apply_liquidity_filter(
    snapshots: list,
    min_adv_usd: float = 10_000_000,
    min_price: float = 5.0,
) -> list:
    """Filter snapshots to liquid, reasonably-priced stocks.

    Args:
        snapshots: List of StockSnapshot objects from the morning scan.
        min_adv_usd: Minimum average daily dollar volume (price × volume).
        min_price: Minimum stock price to exclude penny stocks.

    Returns:
        Filtered list of snapshots passing both criteria.
    """
    result = []
    removed = 0
    for snap in snapshots:
        adv = snap.avg_volume_30d * snap.prev_close
        if snap.prev_close < min_price:
            removed += 1
            continue
        if adv < min_adv_usd:
            removed += 1
            continue
        result.append(snap)

    logger.info(
        f"Liquidity filter: {len(snapshots)} → {len(result)} stocks "
        f"(removed {removed}: price < ${min_price} or ADV < ${min_adv_usd/1e6:.0f}M)"
    )
    return result
