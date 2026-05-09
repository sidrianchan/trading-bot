"""Trend indicators: EMA(20), EMA(50), SMA(200) and the trend-bias classifier.

All functions accept a ``pd.DataFrame`` of OHLCV bars indexed by timestamp
and return ``pd.Series`` aligned to the input index. No state, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

TrendBias = Literal["uptrend", "downtrend", "range"]


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average of ``close`` over ``period`` bars."""
    if period < 1:
        raise ValueError("ema: period must be ≥ 1")
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average of ``close`` over ``period`` bars."""
    if period < 1:
        raise ValueError("sma: period must be ≥ 1")
    return close.rolling(window=period, min_periods=period).mean()


@dataclass(frozen=True)
class TrendSnapshot:
    """Trend context at the most recent bar."""

    bias: TrendBias
    last_close: float
    ema_fast: float
    ema_slow: float
    sma_long: float
    above_long_ma: bool          # close > 200 SMA
    fast_above_slow: bool        # 20 EMA > 50 EMA


def classify(
    bars: pd.DataFrame,
    *,
    ema_fast: int = 20,
    ema_slow: int = 50,
    sma_long: int = 200,
) -> TrendSnapshot | None:
    """Classify the trend on a daily-bar series.

    Rules (matches the spec):

    * **uptrend**   — close > 200 SMA AND 20 EMA > 50 EMA
    * **downtrend** — close < 200 SMA AND 20 EMA < 50 EMA
    * **range**     — anything else (between the two MAs, or MAs crossed
      against the price)

    Returns ``None`` when there is not enough history to evaluate.
    """
    close = bars["close"]
    if len(close) < sma_long:
        return None

    e_fast = ema(close, ema_fast).iloc[-1]
    e_slow = ema(close, ema_slow).iloc[-1]
    s_long = sma(close, sma_long).iloc[-1]
    last = float(close.iloc[-1])

    if pd.isna(e_fast) or pd.isna(e_slow) or pd.isna(s_long):
        return None

    above_long = last > s_long
    fast_above = e_fast > e_slow

    if above_long and fast_above:
        bias: TrendBias = "uptrend"
    elif (not above_long) and (not fast_above):
        bias = "downtrend"
    else:
        bias = "range"

    return TrendSnapshot(
        bias=bias,
        last_close=last,
        ema_fast=float(e_fast),
        ema_slow=float(e_slow),
        sma_long=float(s_long),
        above_long_ma=above_long,
        fast_above_slow=fast_above,
    )
