"""Level quality: how strong is a support zone, and is it holding or breaking?

`signals/price_action/support_resistance.py` produces levels with no notion of
quality — its own comment concedes "weight tracking is unused for now", so a
level touched six times ranks identically to one touched once. That is very
likely why six prior attempts could not, in the words of DECISIONS.md,
"distinguish 'support holding' from 'support breaking'": the engine had no
information with which to make the distinction. This module supplies it.

Three departures from the old module:

1. **Zones are bands, not lines.** Price respects areas; an exact price is an
   artifact of one print. Bands are ATR-scaled so they mean the same thing on a
   $20 stock and a $2000 one.
2. **Reaction-weighted, not touch-counted.** Five touches that each produced a
   0.3-ATR bounce describe a level about to fail. Two touches that each produced
   a 3-ATR bounce describe a level being defended. Touch count alone cannot tell
   these apart.
3. **Round numbers are a bonus, never a source.** The old `round_number_levels`
   emitted every whole dollar and then, because clustering sorted by price and
   kept the first entry, *deleted* real swing pivots that sat just above one.

SEALING — the most important property in this file
--------------------------------------------------
`reaction_atr` looks 10 bars forward. If the strength used to judge a touch
included that touch's own reaction, any study built on it would be predicting the
outcome from the outcome and would fabricate an enormous fake edge.

So event detection (a whole-frame pass) is kept strictly separate from scoring.
`ZoneTrack.strength_as_of(t)` counts a touch only when its full reaction window
closed at or before `t-1`. A touch three bars old is excluded outright, not
partially credited, because its reaction is not yet observable. Never reintroduce
a `.rolling()` or `.shift()` shortcut here: the lookback is event-indexed, not
bar-indexed, so a bar-window shift would silently leak.

Scope: support zones only (the probe is long-only). Resistance is a sign flip,
deliberately not written until it is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from signals.indicators.volatility import atr
from signals.price_action.support_resistance import find_swing_pivots
from signals.price_action.volume_profile import high_volume_nodes, volume_profile

ZoneState = Literal["virgin", "tested", "broken", "reclaimed", "failed_break"]
ZoneOrigin = Literal["pivot", "hvn", "anchored_vwap"]

REACTION_WINDOW = 10          # bars of forward excursion that define a reaction
BAND_ATR_MULT = 0.35          # zone half-width
MIN_TOUCH_SEPARATION = 5      # bars between touches that count separately
MIN_EXCURSION_ATR = 1.0       # price must leave the zone by this much between touches
BREAK_ATR_MULT = 0.5          # close must clear the band by this to be a break
BREAK_VOLUME_RATIO = 1.3      # ...on at least this much of the 20-day average volume
RECLAIM_WINDOW = 5            # bars to close back inside and reclaim
STATE_BONUS_WINDOW = 20       # bars a reclaim/failed-break stays informative
ROUND_LADDERS = (5.0, 10.0, 25.0, 50.0, 100.0)   # NB: no $1 rung, by design
ROUND_TOLERANCE_PCT = 0.0015


@dataclass(frozen=True)
class Touch:
    bar_idx: int
    penetration_atr: float
    reaction_atr: float
    volume_ratio: float

    @property
    def sealed_at(self) -> int:
        """First bar index at which this touch's reaction is fully observable."""
        return self.bar_idx + REACTION_WINDOW


@dataclass(frozen=True)
class Break:
    bar_idx: int
    volume_ratio: float
    clean: bool          # cleared the band on real volume (vs a low-volume poke)
    reclaimed: bool


@dataclass(frozen=True)
class StrengthWeights:
    touch: float = 0.25
    reaction: float = 0.25
    volume: float = 0.20
    confluence: float = 0.15
    recency: float = 0.10
    round_number: float = 0.05
    break_penalty: float = 0.30


WEIGHTS = StrengthWeights()


@dataclass
class ZoneTrack:
    """A support band plus its full event history over the sample.

    The event lists span the whole frame. Anything that scores the zone must go
    through `strength_as_of` / `state_as_of`, which apply the sealing rule.
    """

    lo: float
    hi: float
    anchor: float
    origins: frozenset[ZoneOrigin]
    touches: tuple[Touch, ...] = ()
    breaks: tuple[Break, ...] = ()
    volume_share: float = 0.0
    _states: tuple[tuple[int, ZoneState], ...] = field(default=(), repr=False)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def state_as_of(self, t: int) -> ZoneState:
        """Zone state at bar `t`, from transitions strictly at or before `t`."""
        state: ZoneState = "virgin"
        for idx, s in self._states:
            if idx > t:
                break
            state = s
        return state

    def _last_transition_idx(self, t: int) -> int | None:
        idx = None
        for i, _ in self._states:
            if i > t:
                break
            idx = i
        return idx

    def break_cycles_as_of(self, t: int) -> int:
        """Completed break->reclaim round trips at or before `t`."""
        return sum(1 for b in self.breaks if b.bar_idx <= t and b.clean and b.reclaimed)

    def sealed_touches(self, t: int) -> list[Touch]:
        """Touches whose reaction window closed at or before `t-1`.

        This is the seal. A touch is included only when nothing about it depends
        on information at or after bar `t`.
        """
        return [tc for tc in self.touches if tc.sealed_at <= t - 1]

    def strength_as_of(self, t: int) -> float:
        """Zone strength in [0, 1] using only information available before `t`."""
        sealed = self.sealed_touches(t)
        if not sealed:
            return 0.0

        f_touch = min(len(sealed), 4) / 4.0

        reactions = [tc.reaction_atr for tc in sealed]
        f_reaction = float(np.clip(float(np.median(reactions)) / 2.0, 0.0, 1.0))

        f_volume = float(np.clip(self.volume_share, 0.0, 1.0))
        f_confluence = min(len(self.origins), 3) / 3.0

        bars_since = t - sealed[-1].bar_idx
        f_recency = math.exp(-max(bars_since, 0) / 45.0)

        f_round = 1.0 if _near_round_number(self.anchor) else 0.0

        # Only breaks that are both sealed and never reclaimed count against the
        # zone. A clean break that was reclaimed is a failed break — handled as a
        # positive below, not a penalty.
        n_clean = sum(
            1 for b in self.breaks
            if b.bar_idx <= t - 1 and b.clean and not b.reclaimed
        )

        score = (
            WEIGHTS.touch * f_touch
            + WEIGHTS.reaction * f_reaction
            + WEIGHTS.volume * f_volume
            + WEIGHTS.confluence * f_confluence
            + WEIGHTS.recency * f_recency
            + WEIGHTS.round_number * f_round
            - WEIGHTS.break_penalty * n_clean
        )

        # A zone broken on real volume and then reclaimed, or poked and
        # defended, has proven itself under pressure — the highest-conviction
        # structure here, and something the prior attempts could not represent
        # at all. The bonus is time-limited: a failed break two years ago says
        # nothing about today, and letting the state persist indefinitely would
        # hand a permanent bonus to most zones and flatten the score.
        recent = self._last_transition_idx(t)
        if (
            self.state_as_of(t) in ("reclaimed", "failed_break")
            and recent is not None
            and t - recent <= STATE_BONUS_WINDOW
        ):
            score += 0.10

        return float(np.clip(score, 0.0, 1.0))


def _near_round_number(price: float, ladders=ROUND_LADDERS, tol=ROUND_TOLERANCE_PCT) -> bool:
    for step in ladders:
        nearest = round(price / step) * step
        if nearest > 0 and abs(price - nearest) / price <= tol:
            return True
    return False


def _candidate_anchors(bars: pd.DataFrame, atr_series: pd.Series) -> dict[float, set[ZoneOrigin]]:
    """Seed prices for zones, tagged by which source proposed them."""
    anchors: dict[float, set[ZoneOrigin]] = {}

    def add(price: float, origin: ZoneOrigin) -> None:
        if price is None or not np.isfinite(price) or price <= 0:
            return
        anchors.setdefault(float(price), set()).add(origin)

    highs, lows = find_swing_pivots(bars, window=5)
    for p in lows:
        add(p, "pivot")

    profile = volume_profile(bars, bins=60)
    for node in high_volume_nodes(profile, pct_of_poc=0.70):
        add(node, "hvn")

    for p in _anchored_vwaps(bars):
        add(p, "anchored_vwap")

    return anchors


def _anchored_vwaps(bars: pd.DataFrame) -> list[float]:
    """VWAP anchored at structurally meaningful origins.

    Two anchors: the most recent major swing low, and the highest-volume session
    of the trailing 60 bars. Both are prices large participants actually
    reference, and both are cheap to compute.
    """
    out: list[float] = []
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    vol = bars["volume"]

    _, lows = find_swing_pivots(bars, window=10)
    if lows:
        low_price = lows[-1]
        matches = bars.index[bars["low"] <= low_price * 1.001]
        if len(matches) > 0:
            start = bars.index.get_loc(matches[-1])
            out.append(_vwap_from(typical, vol, start))

    tail = bars.tail(60)
    if not tail.empty and tail["volume"].max() > 0:
        hv_idx = bars.index.get_loc(tail["volume"].idxmax())
        out.append(_vwap_from(typical, vol, hv_idx))

    return [p for p in out if p is not None]


def _vwap_from(typical: pd.Series, vol: pd.Series, start: int) -> float | None:
    t, v = typical.iloc[start:], vol.iloc[start:]
    total = float(v.sum())
    if total <= 0:
        return None
    return float((t * v).sum() / total)


def build_zones(bars: pd.DataFrame, min_touches: int = 2) -> list[ZoneTrack]:
    """Detect support zones and record their full event history.

    The returned tracks carry whole-frame event lists. Scoring must go through
    `strength_as_of` / `state_as_of`, which seal against look-ahead.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"build_zones requires columns {sorted(missing)}")
    if len(bars) < 60:
        return []

    bars = bars.sort_index()
    atr_series = atr(bars, period=20)
    vol_avg = bars["volume"].rolling(20, min_periods=5).mean()

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    atr_arr = atr_series.to_numpy(dtype=float)
    vol_avg_arr = vol_avg.to_numpy(dtype=float)

    profile = volume_profile(bars, bins=60)
    total_profile_volume = float(profile.sum()) if not profile.empty else 0.0

    anchors = _candidate_anchors(bars, atr_series)
    # Band width must scale with price, not be a fixed dollar amount: over a long
    # sample a name can triple, and one global median ATR would make bands far
    # too wide early and too narrow late. Use typical ATR *as a fraction of
    # price* and apply it at each anchor's own level.
    atr_pct = float(np.nanmedian(atr_arr / close))
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return []

    tracks: list[ZoneTrack] = []

    for anchor, origins in _merge_anchors(anchors, atr_pct):
        half = BAND_ATR_MULT * atr_pct * anchor
        lo, hi = anchor - half, anchor + half

        touches, breaks, states = _scan_events(
            lo, hi, high, low, close, volume, atr_arr, vol_avg_arr
        )
        if len(touches) < min_touches:
            continue

        share = 0.0
        if total_profile_volume > 0 and not profile.empty:
            in_band = profile[(profile.index >= lo) & (profile.index <= hi)]
            # Share relative to what a flat profile would put in a band this
            # wide, so wide bands aren't rewarded for being wide.
            expected = total_profile_volume * (hi - lo) / (
                float(profile.index.max() - profile.index.min()) or 1.0
            )
            share = float(in_band.sum() / expected) / 3.0 if expected > 0 else 0.0

        tracks.append(
            ZoneTrack(
                lo=lo, hi=hi, anchor=anchor,
                origins=frozenset(origins),
                touches=tuple(touches),
                breaks=tuple(breaks),
                volume_share=share,
                _states=tuple(states),
            )
        )

    return tracks


def _merge_anchors(
    anchors: dict[float, set[ZoneOrigin]], atr_pct: float
) -> list[tuple[float, set[ZoneOrigin]]]:
    """Collapse nearby candidate prices into one band, unioning their origins.

    Merging rather than discarding is what makes confluence mean anything: when a
    swing pivot, a high-volume node and an anchored VWAP all land in the same
    area, that agreement is the signal. Dropping the later arrivals as
    "duplicates" would silently reduce every zone to a single origin and flatten
    the confluence term to a constant.
    """
    if not anchors:
        return []

    merged: list[tuple[float, set[ZoneOrigin]]] = []
    group_prices: list[float] = []
    group_origins: set[ZoneOrigin] = set()

    for price in sorted(anchors):
        if group_prices:
            centre = sum(group_prices) / len(group_prices)
            if abs(price - centre) > BAND_ATR_MULT * atr_pct * centre:
                merged.append((centre, group_origins))
                group_prices, group_origins = [], set()
        group_prices.append(price)
        group_origins |= anchors[price]

    if group_prices:
        merged.append((sum(group_prices) / len(group_prices), group_origins))
    return merged


def _scan_events(
    lo: float, hi: float,
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    atr_arr: np.ndarray, vol_avg_arr: np.ndarray,
) -> tuple[list[Touch], list[Break], list[tuple[int, ZoneState]]]:
    """Single forward pass recording touches, breaks and state transitions.

    The state machine loops by construction: every state has a defined successor
    for every event, nothing is terminal, and a zone may cycle
    virgin -> tested -> broken -> reclaimed -> tested -> broken indefinitely.
    """
    n = len(close)
    touches: list[Touch] = []
    breaks: list[Break] = []
    states: list[tuple[int, ZoneState]] = []
    state: ZoneState = "virgin"

    last_touch_idx = -10_000
    left_zone_since_touch = True
    pending_break: int | None = None
    # A zone can only break while price is actually engaging it. Without this,
    # a zone left far overhead re-registers a "break" on every subsequent bar
    # for years, which both floods the break list and zeroes the strength score.
    engaged = True

    for i in range(n):
        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        vr = float(volume[i] / vol_avg_arr[i]) if vol_avg_arr[i] and np.isfinite(vol_avg_arr[i]) else 1.0

        overlaps = (low[i] <= hi) and (high[i] >= lo)
        closed_below = close[i] < lo - BREAK_ATR_MULT * a
        closed_inside = lo <= close[i] <= hi

        # --- reclaim: back inside within the window after a break ---
        if pending_break is not None:
            if closed_inside:
                b = breaks[-1]
                breaks[-1] = Break(b.bar_idx, b.volume_ratio, b.clean, reclaimed=True)
                state = "reclaimed"
                states.append((i, state))
                pending_break = None
                left_zone_since_touch = False
                last_touch_idx = i
            elif i - pending_break > RECLAIM_WINDOW:
                pending_break = None

        # Price back at or above the zone re-arms it, so a later failure counts
        # as a genuinely new break. This is what makes the state machine loop.
        if close[i] >= lo:
            engaged = True

        # --- break vs failed break ---
        if closed_below:
            if engaged:
                clean = vr >= BREAK_VOLUME_RATIO
                breaks.append(Break(bar_idx=i, volume_ratio=vr, clean=clean, reclaimed=False))
                state = "broken" if clean else "failed_break"
                states.append((i, state))
                if clean:
                    pending_break = i
                engaged = False
            left_zone_since_touch = True
            continue

        # A wick through that closed back inside is defence, not failure.
        if overlaps and low[i] < lo - BREAK_ATR_MULT * a and closed_inside:
            if state != "failed_break":
                state = "failed_break"
                states.append((i, state))

        # --- touches ---
        if overlaps:
            far_enough = (i - last_touch_idx) >= MIN_TOUCH_SEPARATION
            if far_enough and left_zone_since_touch:
                window_end = min(i + 1 + REACTION_WINDOW, n)
                if window_end > i + 1:
                    reaction = float((np.nanmax(high[i + 1 : window_end]) - close[i]) / a)
                else:
                    reaction = 0.0
                touches.append(
                    Touch(
                        bar_idx=i,
                        penetration_atr=float(max(0.0, (lo - low[i]) / a)),
                        reaction_atr=max(0.0, reaction),
                        volume_ratio=vr,
                    )
                )
                last_touch_idx = i
                left_zone_since_touch = False
                if state in ("virgin", "reclaimed"):
                    state = "tested"
                    states.append((i, state))
        elif low[i] > hi + MIN_EXCURSION_ATR * a:
            # Price travelled a full ATR clear of the zone, so the next visit is
            # a genuinely separate test rather than more of the same chop.
            left_zone_since_touch = True

    return touches, breaks, states
