from __future__ import annotations

import os
from datetime import date

import pandas as pd
import yfinance as yf
from loguru import logger

from data.cache import Cache

_cache = Cache(ttl_hours=12)


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Return adjusted close prices as a DataFrame (dates x tickers).

    source: 'yfinance' for backtesting, 'polygon' for live.
    Always includes SPY for benchmark / trend-filter use.
    """
    all_tickers = sorted(set(tickers) | {"SPY"})
    cache_key = f"prices_{source}_{start}_{end}_{'_'.join(all_tickers[:5])}_n{len(all_tickers)}"

    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    if source == "yfinance":
        df = _fetch_yfinance(all_tickers, start, end)
    elif source == "polygon":
        df = _fetch_polygon(all_tickers, start, end)
    else:
        raise ValueError(f"Unknown source: {source}")

    _cache.set(cache_key, df)
    return df


def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    logger.info(f"Downloading {len(tickers)} tickers from yfinance ({start} → {end})")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers

    # Drop tickers with >20% missing data
    threshold = 0.80
    close = close.loc[:, close.notna().mean() >= threshold]
    close = close.ffill().dropna(how="all")
    logger.info(f"Loaded {close.shape[1]} tickers after quality filter")
    return close


def fetch_intraday_bars(
    tickers: list[str],
    date: str,
    resolution: str = "1Min",
) -> dict[str, pd.DataFrame]:
    """Fetch 1-minute OHLCV bars for a single trading day from Alpaca.

    Returns dict of ticker → DataFrame with columns [open, high, low, close, volume].
    Indexed by timestamp (UTC). Empty DataFrame if data unavailable.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in environment.
    """
    import os
    import time as _time
    from datetime import datetime, timezone

    try:
        from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
        from alpaca.data.requests import StockBarsRequest             # type: ignore
        from alpaca.data.timeframe import TimeFrame                   # type: ignore
    except ImportError:
        logger.error("alpaca-py not installed — cannot fetch intraday bars")
        return {}

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    client = StockHistoricalDataClient(api_key, secret_key)

    start_dt = datetime.fromisoformat(f"{date}T09:30:00").replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(f"{date}T16:00:00").replace(tzinfo=timezone.utc)

    tf = TimeFrame.Minute if resolution == "1Min" else TimeFrame.Minute
    results: dict[str, pd.DataFrame] = {}

    # Batch in chunks of 25 to respect rate limits
    chunk_size = 25
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=tf, start=start_dt, end=end_dt)
            bars = client.get_stock_bars(req)
            bar_dict = bars.data if hasattr(bars, "data") else bars
            for ticker in chunk:
                try:
                    raw = bar_dict.get(ticker, [])
                    if not raw:
                        continue
                    rows = [{"open": b.open, "high": b.high, "low": b.low,
                             "close": b.close, "volume": b.volume,
                             "timestamp": b.timestamp} for b in raw]
                    df = pd.DataFrame(rows).set_index("timestamp")
                    results[ticker] = df
                except (KeyError, AttributeError):
                    pass
        except Exception as exc:
            logger.warning(f"Intraday bar fetch failed for chunk {chunk[:3]}…: {exc}")
        if i + chunk_size < len(tickers):
            _time.sleep(0.3)

    logger.info(f"Fetched intraday bars for {len(results)}/{len(tickers)} tickers on {date}")
    return results


def fetch_intraday_bars_range(
    tickers: list[str],
    start_date: str,
    end_date: str,
    resolution: str = "1Min",
    cache_dir: str = "data/cache/intraday",
) -> dict[str, pd.DataFrame]:
    """Fetch 1-min bars for multiple tickers over a full date range from Alpaca.

    Caches each ticker's bars to parquet so repeated backtest runs skip re-fetching.
    Returns dict of ticker → DataFrame indexed by timestamp.
    """
    import time as _time
    from datetime import datetime, timezone
    from pathlib import Path

    try:
        from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
        from alpaca.data.requests import StockBarsRequest             # type: ignore
        from alpaca.data.timeframe import TimeFrame                   # type: ignore
    except ImportError:
        logger.error("alpaca-py not installed — cannot fetch intraday bars")
        return {}

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    client = StockHistoricalDataClient(api_key, secret_key)

    start_dt = datetime.fromisoformat(f"{start_date}T09:30:00").replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(f"{end_date}T16:00:00").replace(tzinfo=timezone.utc)
    tf = TimeFrame.Minute

    results: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        pq = cache_path / f"{ticker}_{start_date}_{end_date}.parquet"
        if pq.exists():
            try:
                results[ticker] = pd.read_parquet(pq)
                continue
            except Exception:
                pass
        to_fetch.append(ticker)

    if not to_fetch:
        logger.info(f"All {len(results)} tickers loaded from parquet cache")
        return results

    logger.info(f"Fetching range bars for {len(to_fetch)} tickers ({start_date} → {end_date})")

    # Small chunks for multi-day ranges (larger payloads per ticker)
    chunk_size = 5
    for i in range(0, len(to_fetch), chunk_size):
        chunk = to_fetch[i : i + chunk_size]
        try:
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=tf, start=start_dt, end=end_dt)
            bars = client.get_stock_bars(req)
            bar_dict = bars.data if hasattr(bars, "data") else bars
            for ticker in chunk:
                raw = bar_dict.get(ticker, [])
                if not raw:
                    continue
                rows = [{"open": b.open, "high": b.high, "low": b.low,
                         "close": b.close, "volume": b.volume,
                         "timestamp": b.timestamp} for b in raw]
                df = pd.DataFrame(rows).set_index("timestamp")
                results[ticker] = df
                pq = cache_path / f"{ticker}_{start_date}_{end_date}.parquet"
                try:
                    df.to_parquet(pq)
                except Exception as exc:
                    logger.warning(f"Parquet cache write failed for {ticker}: {exc}")
        except Exception as exc:
            logger.warning(f"Range bar fetch failed for chunk {chunk[:3]}…: {exc} — retrying individually")
            _time.sleep(0.3)
            for ticker in chunk:
                try:
                    req1 = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=tf, start=start_dt, end=end_dt)
                    b1 = client.get_stock_bars(req1)
                    bd1 = b1.data if hasattr(b1, "data") else b1
                    raw = bd1.get(ticker, [])
                    if not raw:
                        continue
                    rows = [{"open": b.open, "high": b.high, "low": b.low,
                             "close": b.close, "volume": b.volume,
                             "timestamp": b.timestamp} for b in raw]
                    df = pd.DataFrame(rows).set_index("timestamp")
                    results[ticker] = df
                    pq = cache_path / f"{ticker}_{start_date}_{end_date}.parquet"
                    try:
                        df.to_parquet(pq)
                    except Exception:
                        pass
                except Exception as exc2:
                    logger.debug(f"Skipping {ticker}: {exc2}")
        if i + chunk_size < len(to_fetch):
            _time.sleep(0.5)

    logger.info(f"Fetched range bars for {len(results)}/{len(tickers)} tickers")
    return results


def _fetch_polygon(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    from polygon import RESTClient  # type: ignore

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise EnvironmentError("POLYGON_API_KEY not set in environment")

    client = RESTClient(api_key)
    frames: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            bars = client.get_aggs(ticker, 1, "day", start, end, adjusted=True, limit=50000)
            if not bars:
                continue
            s = pd.Series(
                {pd.Timestamp(b.timestamp, unit="ms"): b.close for b in bars},
                name=ticker,
            )
            frames[ticker] = s
        except Exception as exc:
            logger.warning(f"Polygon fetch failed for {ticker}: {exc}")

    if not frames:
        raise RuntimeError("No data returned from Polygon")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()
