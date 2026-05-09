"""Support / resistance level detection.

Three sources of levels are produced and merged:

1. **Swing pivots** — local highs and lows on the supplied bar series. A bar
   qualifies as a swing high if its high is strictly greater than the highs
   of the ``window`` bars on each side; symmetric for swing lows.

2. **Round numbers** — psychological levels (``$1, $5, $10, $25, $50, $100``
   and their multiples) within the current price range.

3. **Reference levels** — prior day high/low and prior week high/low. These
   require the bar series to span at least one full prior session/week.

Nearby levels are merged when they fall within ``cluster_tolerance_pct`` of
each other to avoid scoring the same wall twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

LevelKind = Literal["swing_high", "swing_low", "round", "pdh", "pdl", "pwh", "pwl"]


@dataclass(frozen=True)
class Level:
    price: float
    kind: LevelKind
    role: Literal["support", "resistance"]


@dataclass(frozen=True)
class LevelMap:
    levels: list[Level]
    last_price: float

    def near(self, price: float, tolerance_pct: float = 0.005) -> list[Level]:
        """Levels within ``tolerance_pct`` of ``price``."""
        if price <= 0:
            return []
        return [lvl for lvl in self.levels if abs(lvl.price - price) / price <= tolerance_pct]

    def nearest(self, price: float) -> Level | None:
        if not self.levels:
            return None
        return min(self.levels, key=lambda lvl: abs(lvl.price - price))


def find_swing_pivots(
    bars: pd.DataFrame, *, window: int = 5
) -> tuple[list[float], list[float]]:
    """Return (swing_highs, swing_lows). ``window`` bars on each side.

    Excludes the trailing ``window`` bars so we never label an in-progress
    move as a confirmed pivot.
    """
    if len(bars) < 2 * window + 1:
        return [], []
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    n = len(bars)

    highs: list[float] = []
    lows: list[float] = []
    for i in range(window, n - window):
        seg_h = high[i - window : i + window + 1]
        seg_l = low[i - window : i + window + 1]
        if high[i] == seg_h.max() and (seg_h == high[i]).sum() == 1:
            highs.append(float(high[i]))
        if low[i] == seg_l.min() and (seg_l == low[i]).sum() == 1:
            lows.append(float(low[i]))
    return highs, lows


def round_number_levels(
    price_min: float, price_max: float, ladders: list[int] | None = None
) -> list[float]:
    """Generate psychological round-number levels within [price_min, price_max].

    ``ladders`` are step sizes; e.g. ``[1, 5, 10, 25, 50, 100]`` produces
    every $1, $5, $10, $25, $50, $100 within range. Output is deduplicated
    and sorted.
    """
    ladders = ladders or [1, 5, 10, 25, 50, 100]
    out: set[float] = set()
    for step in ladders:
        if step <= 0:
            continue
        start = (int(price_min) // step) * step
        if start < price_min:
            start += step
        v = start
        while v <= price_max:
            out.add(float(v))
            v += step
    return sorted(out)


def prior_session_levels(daily_bars: pd.DataFrame) -> dict[str, float]:
    """Extract prior-day H/L and prior-week H/L from a *daily* bar series.

    The most recent bar in ``daily_bars`` is treated as the "current" bar;
    PDH/PDL come from the bar immediately before it; PWH/PWL come from the
    most recent completed calendar week excluding the current bar's week.
    """
    out: dict[str, float] = {}
    if len(daily_bars) < 2:
        return out

    prior = daily_bars.iloc[-2]
    out["pdh"] = float(prior["high"])
    out["pdl"] = float(prior["low"])

    if not isinstance(daily_bars.index, pd.DatetimeIndex):
        return out
    iso_year = daily_bars.index.isocalendar().year
    iso_week = daily_bars.index.isocalendar().week
    current_year = int(iso_year.iloc[-1])
    current_week = int(iso_week.iloc[-1])
    prev_week_mask = ~((iso_year == current_year) & (iso_week == current_week))
    prev_week = daily_bars[prev_week_mask]
    if not prev_week.empty:
        last_year = int(iso_year[prev_week_mask].iloc[-1])
        last_week = int(iso_week[prev_week_mask].iloc[-1])
        wk_mask = (iso_year[prev_week_mask] == last_year) & (iso_week[prev_week_mask] == last_week)
        wk = prev_week[wk_mask.to_numpy()]
        if not wk.empty:
            out["pwh"] = float(wk["high"].max())
            out["pwl"] = float(wk["low"].min())
    return out


def build_level_map(
    bars: pd.DataFrame,
    *,
    swing_window: int = 5,
    round_ladders: list[int] | None = None,
    cluster_tolerance_pct: float = 0.005,
    daily_for_reference: pd.DataFrame | None = None,
) -> LevelMap:
    """Combine swing pivots, round numbers, and reference levels into a
    de-clustered ``LevelMap``.

    ``role`` is assigned by comparing each level to the latest close: levels
    above the price are *resistance*, levels below are *support*.
    """
    if bars.empty:
        return LevelMap(levels=[], last_price=0.0)

    last_price = float(bars["close"].iloc[-1])

    highs, lows = find_swing_pivots(bars, window=swing_window)
    raw: list[tuple[float, LevelKind]] = []
    raw.extend((p, "swing_high") for p in highs)
    raw.extend((p, "swing_low") for p in lows)

    pmin = float(bars["low"].min())
    pmax = float(bars["high"].max())
    for r in round_number_levels(pmin, pmax, round_ladders):
        raw.append((r, "round"))

    if daily_for_reference is not None and not daily_for_reference.empty:
        prior = prior_session_levels(daily_for_reference)
        for k, v in prior.items():
            raw.append((v, k))  # type: ignore[arg-type]

    # Cluster nearby levels (keep the first; weight tracking is unused for now)
    raw.sort()
    clustered: list[tuple[float, LevelKind]] = []
    for price, kind in raw:
        if not clustered:
            clustered.append((price, kind))
            continue
        last = clustered[-1][0]
        if last == 0 or abs(price - last) / last > cluster_tolerance_pct:
            clustered.append((price, kind))

    levels = [
        Level(
            price=p,
            kind=k,
            role="resistance" if p >= last_price else "support",
        )
        for p, k in clustered
    ]
    return LevelMap(levels=levels, last_price=last_price)
