"""Candlestick pattern detectors.

Each detector inspects only the most recent N bars (1-3 depending on the
pattern). Returns a small dataclass with ``triggered`` plus the metric used
in the decision so callers can score and explain the setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

PatternName = Literal[
    "hammer",
    "bullish_engulfing",
    "morning_star",
    "bullish_pin_bar",
    "shooting_star",
    "bearish_engulfing",
    "evening_star",
    "bearish_pin_bar",
]
Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class CandlePattern:
    name: PatternName
    direction: Direction
    triggered: bool
    body_ratio: float = 0.0
    detail: str = ""


def _ohlc(row: pd.Series) -> tuple[float, float, float, float]:
    return float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])


def _body(o: float, c: float) -> float:
    return abs(c - o)


def _range(h: float, l: float) -> float:
    return max(h - l, 1e-12)


def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l


# ── Bullish patterns ───────────────────────────────────────────────────────


def hammer(bars: pd.DataFrame, *, body_ratio_threshold: float = 0.3) -> CandlePattern:
    """Single-bar hammer: small body at the top, long lower wick (≥ 2× body)."""
    if len(bars) < 1:
        return CandlePattern("hammer", "bullish", False)
    o, h, l, c = _ohlc(bars.iloc[-1])
    body = _body(o, c)
    rng = _range(h, l)
    body_ratio = body / rng
    lower_wick = _lower_wick(o, l, c)
    upper_wick = _upper_wick(o, h, c)
    triggered = (
        body_ratio <= body_ratio_threshold
        and lower_wick >= 2 * body
        and (upper_wick / rng) <= 0.15
    )
    return CandlePattern("hammer", "bullish", triggered, body_ratio, "lower-wick rejection")


def bullish_engulfing(bars: pd.DataFrame) -> CandlePattern:
    """Two-bar pattern: red candle followed by a green candle whose real body
    fully engulfs the prior body."""
    if len(bars) < 2:
        return CandlePattern("bullish_engulfing", "bullish", False)
    o1, _, _, c1 = _ohlc(bars.iloc[-2])
    o2, _, _, c2 = _ohlc(bars.iloc[-1])
    triggered = c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1
    return CandlePattern("bullish_engulfing", "bullish", triggered)


def morning_star(bars: pd.DataFrame) -> CandlePattern:
    """Three-bar reversal: long red, small body (gap down OK), long green
    closing back into the body of bar 1."""
    if len(bars) < 3:
        return CandlePattern("morning_star", "bullish", False)
    o1, _, _, c1 = _ohlc(bars.iloc[-3])
    o2, h2, l2, c2 = _ohlc(bars.iloc[-2])
    o3, _, _, c3 = _ohlc(bars.iloc[-1])
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    body3 = _body(o3, c3)
    triggered = (
        c1 < o1
        and body1 > 0
        and body2 < 0.5 * body1
        and c3 > o3
        and body3 > 0.5 * body1
        and c3 >= (o1 + c1) / 2
    )
    return CandlePattern("morning_star", "bullish", triggered)


def bullish_pin_bar(bars: pd.DataFrame, *, body_ratio_threshold: float = 0.3) -> CandlePattern:
    """Pin bar with rejection of lower prices: long lower wick, body in upper
    third of the range, regardless of close color."""
    if len(bars) < 1:
        return CandlePattern("bullish_pin_bar", "bullish", False)
    o, h, l, c = _ohlc(bars.iloc[-1])
    rng = _range(h, l)
    body = _body(o, c)
    body_ratio = body / rng
    lower_wick = _lower_wick(o, l, c)
    body_top = max(o, c)
    triggered = (
        body_ratio <= body_ratio_threshold
        and lower_wick >= 2 * body
        and (body_top - l) / rng >= 2 / 3
    )
    return CandlePattern("bullish_pin_bar", "bullish", triggered, body_ratio)


# ── Bearish patterns ───────────────────────────────────────────────────────


def shooting_star(bars: pd.DataFrame, *, body_ratio_threshold: float = 0.3) -> CandlePattern:
    """Single-bar shooting star: small body at the bottom, long upper wick."""
    if len(bars) < 1:
        return CandlePattern("shooting_star", "bearish", False)
    o, h, l, c = _ohlc(bars.iloc[-1])
    body = _body(o, c)
    rng = _range(h, l)
    body_ratio = body / rng
    upper_wick = _upper_wick(o, h, c)
    lower_wick = _lower_wick(o, l, c)
    triggered = (
        body_ratio <= body_ratio_threshold
        and upper_wick >= 2 * body
        and (lower_wick / rng) <= 0.15
    )
    return CandlePattern("shooting_star", "bearish", triggered, body_ratio, "upper-wick rejection")


def bearish_engulfing(bars: pd.DataFrame) -> CandlePattern:
    """Green candle followed by a red candle whose real body fully engulfs
    the prior body."""
    if len(bars) < 2:
        return CandlePattern("bearish_engulfing", "bearish", False)
    o1, _, _, c1 = _ohlc(bars.iloc[-2])
    o2, _, _, c2 = _ohlc(bars.iloc[-1])
    triggered = c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1
    return CandlePattern("bearish_engulfing", "bearish", triggered)


def evening_star(bars: pd.DataFrame) -> CandlePattern:
    """Three-bar reversal: long green, small body, long red closing back into
    body of bar 1."""
    if len(bars) < 3:
        return CandlePattern("evening_star", "bearish", False)
    o1, _, _, c1 = _ohlc(bars.iloc[-3])
    o2, _, _, c2 = _ohlc(bars.iloc[-2])
    o3, _, _, c3 = _ohlc(bars.iloc[-1])
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    body3 = _body(o3, c3)
    triggered = (
        c1 > o1
        and body1 > 0
        and body2 < 0.5 * body1
        and c3 < o3
        and body3 > 0.5 * body1
        and c3 <= (o1 + c1) / 2
    )
    return CandlePattern("evening_star", "bearish", triggered)


def bearish_pin_bar(bars: pd.DataFrame, *, body_ratio_threshold: float = 0.3) -> CandlePattern:
    """Pin bar with rejection of higher prices: long upper wick, body in
    lower third of the range."""
    if len(bars) < 1:
        return CandlePattern("bearish_pin_bar", "bearish", False)
    o, h, l, c = _ohlc(bars.iloc[-1])
    rng = _range(h, l)
    body = _body(o, c)
    body_ratio = body / rng
    upper_wick = _upper_wick(o, h, c)
    body_bottom = min(o, c)
    triggered = (
        body_ratio <= body_ratio_threshold
        and upper_wick >= 2 * body
        and (h - body_bottom) / rng >= 2 / 3
    )
    return CandlePattern("bearish_pin_bar", "bearish", triggered, body_ratio)


# ── Aggregation ────────────────────────────────────────────────────────────


_BULLISH_DETECTORS = (hammer, bullish_engulfing, morning_star, bullish_pin_bar)
_BEARISH_DETECTORS = (shooting_star, bearish_engulfing, evening_star, bearish_pin_bar)


def detect_all(
    bars: pd.DataFrame, *, body_ratio_threshold: float = 0.3
) -> list[CandlePattern]:
    """Run every detector on ``bars`` and return only triggered patterns.

    Detectors that accept ``body_ratio_threshold`` receive it; the others
    are called with their default kwargs.
    """
    out: list[CandlePattern] = []
    for fn in (hammer, bullish_pin_bar, shooting_star, bearish_pin_bar):
        p = fn(bars, body_ratio_threshold=body_ratio_threshold)
        if p.triggered:
            out.append(p)
    for fn in (bullish_engulfing, morning_star, bearish_engulfing, evening_star):
        p = fn(bars)
        if p.triggered:
            out.append(p)
    return out
