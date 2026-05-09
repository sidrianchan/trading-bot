"""Volatility indicators: Bollinger Bands(20, 2σ), ATR(14), squeeze detector."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BollingerBands:
    middle: pd.Series      # SMA(period)
    upper: pd.Series       # SMA + std * std_dev
    lower: pd.Series       # SMA - std * std_dev


def bollinger_bands(close: pd.Series, *, period: int = 20, std_dev: float = 2.0) -> BollingerBands:
    """Classic Bollinger Bands."""
    if period < 2:
        raise ValueError("bollinger: period must be ≥ 2")
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    return BollingerBands(middle=mid, upper=mid + std_dev * std, lower=mid - std_dev * std)


def squeeze(bands: BollingerBands, close: pd.Series, *, max_width_pct: float = 0.01) -> bool:
    """True if the latest bar shows a "squeeze" — the band width as a fraction
    of price is below ``max_width_pct`` (default 1%)."""
    if bands.upper.dropna().empty:
        return False
    width = float(bands.upper.iloc[-1] - bands.lower.iloc[-1])
    last_close = float(close.iloc[-1])
    if last_close <= 0:
        return False
    return (width / last_close) < max_width_pct


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range over ``period`` bars (Wilder smoothing)."""
    if period < 1:
        raise ValueError("atr: period must be ≥ 1")
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def at_upper_band(bands: BollingerBands, close: pd.Series, *, tolerance_pct: float = 0.002) -> bool:
    """Latest close is touching/above the upper band (within ``tolerance_pct``)."""
    if bands.upper.dropna().empty:
        return False
    return float(close.iloc[-1]) >= float(bands.upper.iloc[-1]) * (1 - tolerance_pct)


def at_lower_band(bands: BollingerBands, close: pd.Series, *, tolerance_pct: float = 0.002) -> bool:
    """Latest close is touching/below the lower band (within ``tolerance_pct``)."""
    if bands.lower.dropna().empty:
        return False
    return float(close.iloc[-1]) <= float(bands.lower.iloc[-1]) * (1 + tolerance_pct)
