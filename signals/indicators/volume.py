"""Volume indicators: OBV (On-Balance Volume), volume confirmation ratio."""
from __future__ import annotations

import pandas as pd


def obv(bars: pd.DataFrame) -> pd.Series:
    """On-Balance Volume cumulative series.

    Each bar's volume is added when close > prior close, subtracted when
    close < prior close, and skipped (no change) on flat closes.
    """
    close = bars["close"]
    direction = pd.Series(0, index=close.index, dtype="float64")
    diff = close.diff()
    direction[diff > 0] = 1.0
    direction[diff < 0] = -1.0
    return (direction * bars["volume"].astype("float64")).cumsum()


def obv_breaking_out(obv_series: pd.Series, *, lookback: int = 50) -> bool:
    """True when the latest OBV exceeds its prior ``lookback``-bar maximum.

    Used as a "smart money" confirmation for price breakouts: requires the
    cumulative volume flow to also be making a new high simultaneously.
    """
    if len(obv_series) < lookback + 1:
        return False
    last = float(obv_series.iloc[-1])
    prior_max = float(obv_series.iloc[-lookback - 1 : -1].max())
    return last > prior_max


def volume_ratio(bars: pd.DataFrame, *, period: int = 20) -> float:
    """Latest bar's volume divided by the rolling ``period``-bar average."""
    if len(bars) < period + 1:
        return 0.0
    avg = float(bars["volume"].iloc[-period - 1 : -1].mean())
    if avg <= 0:
        return 0.0
    return float(bars["volume"].iloc[-1]) / avg


def has_volume_confirmation(bars: pd.DataFrame, *, period: int = 20, multiple: float = 1.5) -> bool:
    """Latest volume ≥ ``multiple`` × the prior ``period``-bar average."""
    return volume_ratio(bars, period=period) >= multiple
