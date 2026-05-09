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


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — Wilder's trend-strength measure.

    Returns a series in [0, 100]. ADX < 20 indicates a non-trending /
    range-bound market; ADX > 25 indicates a strong trend.

    Computed from the directional movement system:
        +DM = max(high - prev_high, 0) when high - prev_high > prev_low - low else 0
        -DM = max(prev_low - low, 0)  when prev_low - low > high - prev_high else 0
        TR  = max(h-l, |h-prev_c|, |l-prev_c|)
        +DI = 100 × ema(+DM) / ema(TR)
        -DI = 100 × ema(-DM) / ema(TR)
        DX  = 100 × |+DI − −DI| / (+DI + −DI)
        ADX = ema(DX)
    All EMAs use Wilder smoothing (alpha = 1/period).
    """
    if period < 2:
        raise ValueError("adx: period must be ≥ 2")
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_w
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_w
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


@dataclass(frozen=True)
class TrendSnapshot:
    """Trend context at the most recent bar."""

    bias: TrendBias
    last_close: float
    ema_fast: float
    ema_slow: float
    sma_long: float
    adx: float                   # ADX(14) — trend-strength gauge
    above_long_ma: bool          # close > 200 SMA
    fast_above_slow: bool        # 20 EMA > 50 EMA


def classify(
    bars: pd.DataFrame,
    *,
    ema_fast: int = 20,
    ema_slow: int = 50,
    sma_long: int = 200,
    adx_period: int = 14,
    adx_min: float = 20.0,
) -> TrendSnapshot | None:
    """Classify the trend on a daily-bar series.

    Rules:

    * **uptrend**   — close > 200 SMA AND 20 EMA > 50 EMA AND ADX ≥ ``adx_min``
    * **downtrend** — close < 200 SMA AND 20 EMA < 50 EMA AND ADX ≥ ``adx_min``
    * **range**     — everything else, including names that satisfy the MA
      stack but lack the directional strength (ADX < ``adx_min``)

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

    adx_series = adx(bars, period=adx_period)
    if adx_series.dropna().empty:
        return None
    adx_now = float(adx_series.iloc[-1])

    above_long = last > s_long
    fast_above = e_fast > e_slow
    strong_trend = adx_now >= adx_min

    if above_long and fast_above and strong_trend:
        bias: TrendBias = "uptrend"
    elif (not above_long) and (not fast_above) and strong_trend:
        bias = "downtrend"
    else:
        bias = "range"

    return TrendSnapshot(
        bias=bias,
        last_close=last,
        ema_fast=float(e_fast),
        ema_slow=float(e_slow),
        sma_long=float(s_long),
        adx=adx_now,
        above_long_ma=above_long,
        fast_above_slow=fast_above,
    )
