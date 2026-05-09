"""Momentum indicators: RSI(14), MACD(12/26/9) and divergence detection."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on ``close``.

    Returns a series in [0, 100]. The first ``period`` rows are NaN.
    """
    if period < 2:
        raise ValueError("rsi: period must be ≥ 2")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0)  # all gains => RSI = 100
    return out


@dataclass(frozen=True)
class MACDValues:
    macd: pd.Series        # 12-EMA − 26-EMA
    signal: pd.Series      # 9-EMA of MACD
    histogram: pd.Series   # macd − signal


def macd(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> MACDValues:
    """Standard MACD on ``close``."""
    if fast >= slow:
        raise ValueError("macd: fast period must be < slow period")
    ef = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    es = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ef - es
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    return MACDValues(macd=macd_line, signal=signal_line, histogram=macd_line - signal_line)


def macd_cross(values: MACDValues) -> str | None:
    """Return ``"bull"`` if MACD just crossed above signal on the last bar,
    ``"bear"`` if just crossed below, otherwise ``None``."""
    h = values.histogram.dropna()
    if len(h) < 2:
        return None
    prev, last = h.iloc[-2], h.iloc[-1]
    if prev <= 0 < last:
        return "bull"
    if prev >= 0 > last:
        return "bear"
    return None


def histogram_expanding(values: MACDValues, lookback: int = 3) -> bool:
    """True if the |histogram| is strictly increasing across ``lookback`` bars
    AND keeps the same sign (momentum strengthening in one direction)."""
    h = values.histogram.dropna()
    if len(h) < lookback:
        return False
    tail = h.tail(lookback)
    if (tail > 0).all():
        return tail.is_monotonic_increasing
    if (tail < 0).all():
        return tail.is_monotonic_decreasing
    return False


def bearish_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 20) -> bool:
    """Price makes a higher high but RSI makes a lower high within ``lookback``.

    Detects classic bearish divergence: signals a reversal warning at resistance.
    Compares the most recent price peak with the previous price peak.
    """
    return _divergence(close, rsi_series, lookback, kind="bearish")


def bullish_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 20) -> bool:
    """Price makes a lower low but RSI makes a higher low within ``lookback``."""
    return _divergence(close, rsi_series, lookback, kind="bullish")


def _divergence(
    close: pd.Series,
    rsi_series: pd.Series,
    lookback: int,
    *,
    kind: str,
) -> bool:
    if len(close) < lookback or rsi_series.dropna().shape[0] < lookback:
        return False

    window_close = close.tail(lookback)
    window_rsi = rsi_series.tail(lookback)

    if kind == "bearish":
        # Most recent high vs prior high in the lookback window
        peaks = window_close.nlargest(2)
        if len(peaks) < 2:
            return False
        i_recent, i_prior = peaks.index[0], peaks.index[1]
        if i_recent <= i_prior:
            return False
        price_higher = window_close.loc[i_recent] > window_close.loc[i_prior]
        rsi_lower = window_rsi.loc[i_recent] < window_rsi.loc[i_prior]
        return bool(price_higher and rsi_lower)

    # bullish
    troughs = window_close.nsmallest(2)
    if len(troughs) < 2:
        return False
    i_recent, i_prior = troughs.index[0], troughs.index[1]
    if i_recent <= i_prior:
        return False
    price_lower = window_close.loc[i_recent] < window_close.loc[i_prior]
    rsi_higher = window_rsi.loc[i_recent] > window_rsi.loc[i_prior]
    return bool(price_lower and rsi_higher)
