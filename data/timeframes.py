"""Multi-timeframe resampling helpers.

The TA stack reads three views of the same underlying 1-minute Alpaca data:

* daily       — fetched separately as adjusted-close daily bars (handled in
  ``data.market``); this module does NOT resample to daily.
* 1-hour      — aggregated from 1-min via :func:`resample`.
* 15-minute   — aggregated from 1-min via :func:`resample`.

All resampled bars are right-edge labeled (the timestamp is the bar's *close*
time) and the trailing incomplete bar is dropped to prevent lookahead.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

Timeframe = Literal["15min", "1h"]

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample(bars_1min: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate 1-minute OHLCV bars to a higher timeframe.

    Args:
        bars_1min: DataFrame with columns ``[open, high, low, close, volume]``
            indexed by a ``DatetimeIndex``. Index may be tz-aware; the
            returned frame preserves the same tz.
        timeframe: ``"15T"`` (15 minutes) or ``"1H"`` (1 hour).

    Returns:
        Aggregated bars, right-edge labeled, trailing partial bar dropped.
        Empty DataFrame if input is empty.
    """
    if bars_1min.empty:
        return bars_1min.copy()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars_1min.columns)
    if missing:
        raise ValueError(f"resample: missing columns {sorted(missing)}")

    out = bars_1min.resample(timeframe, label="right", closed="left").agg(_AGG)
    out = out.dropna(subset=["open", "high", "low", "close"])

    # Drop trailing partial bar: a complete bar labeled T contains the full
    # interval [T - freq, T) of 1-min source bars; if the last 1-min bar is
    # earlier than (T - 1min), this bin is incomplete.
    if len(out) > 0:
        last_src = bars_1min.index.max()
        offset = pd.tseries.frequencies.to_offset(timeframe)
        last_complete_close = (last_src + pd.Timedelta(minutes=1)).floor(offset)
        if last_complete_close < out.index[-1]:
            out = out.iloc[:-1]
    return out


def market_hours_only(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular US equity market hours: 09:30–16:00 ET.

    Bars indexed in UTC are translated to America/New_York for the comparison
    and returned in their original timezone. tz-naive input is treated as
    already-ET so callers can use either convention.
    """
    if bars.empty:
        return bars.copy()
    idx = bars.index
    if idx.tz is not None:
        idx_et = idx.tz_convert("America/New_York")
    else:
        idx_et = idx
    h, m = idx_et.hour, idx_et.minute
    after_open = (h > 9) | ((h == 9) & (m >= 30))
    before_close = (h < 16) | ((h == 16) & (m == 0))
    return bars[after_open & before_close]
