"""Tests for the level quality model and volume profile.

The look-ahead guards are the most important tests in this file. The strength
score is built from a forward-looking reaction measurement, so a leak there
would fabricate an edge in the calibration study and every result downstream of
it would be worthless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.price_action.level_quality import (
    REACTION_WINDOW,
    Break,
    Touch,
    ZoneTrack,
    _scan_events,
    build_zones,
)
from signals.price_action.volume_profile import high_volume_nodes, poc, volume_profile


def _zone(touches=(), breaks=(), states=(), **kw) -> ZoneTrack:
    return ZoneTrack(
        lo=kw.get("lo", 100.0), hi=kw.get("hi", 101.0), anchor=kw.get("anchor", 100.5),
        origins=frozenset(kw.get("origins", {"pivot"})),
        touches=tuple(touches), breaks=tuple(breaks),
        volume_share=kw.get("volume_share", 0.5), _states=tuple(states),
    )


def _touch(bar_idx: int, reaction: float = 2.0) -> Touch:
    return Touch(bar_idx=bar_idx, penetration_atr=0.2, reaction_atr=reaction, volume_ratio=1.0)


# ── Look-ahead sealing ──────────────────────────────────────────────────


def test_future_touches_cannot_change_past_strength():
    """The core invariant: strength_as_of(t) is blind to anything at or after t.

    Appending later events must leave an earlier score bit-identical. If this
    fails, the calibration study is predicting outcomes from outcomes.
    """
    early = [_touch(0), _touch(20)]
    zone_then = _zone(touches=early)
    t = 40
    before = zone_then.strength_as_of(t)

    # Same zone, but the future turned out spectacularly.
    zone_now = _zone(touches=early + [_touch(45, reaction=9.0), _touch(60, reaction=9.0)])
    assert zone_now.strength_as_of(t) == before


def test_touch_inside_its_reaction_window_contributes_nothing():
    """A touch whose forward window has not closed is excluded outright."""
    t = 30
    unresolved = _touch(t - 3)          # reaction not yet observable at t
    assert unresolved.sealed_at > t - 1
    assert _zone(touches=[unresolved]).strength_as_of(t) == 0.0

    resolved = _touch(t - REACTION_WINDOW - 1)
    assert resolved.sealed_at <= t - 1
    assert _zone(touches=[resolved]).strength_as_of(t) > 0.0


def test_break_after_t_does_not_penalise_strength_at_t():
    at_t = _zone(touches=[_touch(0)]).strength_as_of(40)
    with_future_break = _zone(
        touches=[_touch(0)],
        breaks=[Break(bar_idx=60, volume_ratio=2.0, clean=True, reclaimed=False)],
    ).strength_as_of(40)
    assert with_future_break == at_t


# ── The core modelling claim ────────────────────────────────────────────


def test_strong_reactions_beat_many_weak_touches():
    """Two touches that produced big bounces outrank five that barely moved.

    This is the whole reason the model exists — the previous implementation
    counted touches and could not tell these two levels apart.
    """
    decisive = _zone(touches=[_touch(0, reaction=3.0), _touch(20, reaction=3.0)])
    feeble = _zone(touches=[_touch(i * 12, reaction=0.15) for i in range(5)])
    t = 200
    assert decisive.strength_as_of(t) > feeble.strength_as_of(t)


def test_unreclaimed_clean_breaks_reduce_strength():
    base = _zone(touches=[_touch(0), _touch(20)])
    broken = _zone(
        touches=[_touch(0), _touch(20)],
        breaks=[Break(bar_idx=30, volume_ratio=2.0, clean=True, reclaimed=False)],
    )
    assert broken.strength_as_of(60) < base.strength_as_of(60)


def test_state_bonus_expires():
    """A failed break two years ago says nothing about today."""
    touches = [_touch(0), _touch(20)]
    states = [(35, "failed_break")]
    fresh = _zone(touches=touches, states=states).strength_as_of(45)
    stale = _zone(touches=touches, states=states).strength_as_of(400)
    assert fresh > stale


# ── State machine ───────────────────────────────────────────────────────


def _arrays(closes, lows=None, highs=None, vol_ratio=1.0, atr_val=2.0):
    n = len(closes)
    close = np.array(closes, dtype=float)
    low = np.array(lows if lows is not None else closes, dtype=float)
    high = np.array(highs if highs is not None else closes, dtype=float)
    vol_avg = np.full(n, 1_000_000.0)
    volume = vol_avg * vol_ratio
    return high, low, close, volume, np.full(n, atr_val), vol_avg


def test_low_volume_breakdown_is_a_failed_break_not_a_break():
    # lo=100, hi=101, atr=2 -> break threshold is a close below 99
    closes = [105] * 6 + [98] + [105] * 5
    lows = [104] * 6 + [97] + [104] * 5
    args = _arrays(closes, lows=lows, highs=closes, vol_ratio=0.8)
    _, breaks, states = _scan_events(100.0, 101.0, *args)
    assert len(breaks) == 1 and breaks[0].clean is False
    assert any(s == "failed_break" for _, s in states)


def test_high_volume_breakdown_is_a_clean_break():
    closes = [105] * 6 + [98] + [97] * 5
    lows = [104] * 6 + [97] + [96] * 5
    args = _arrays(closes, lows=lows, highs=closes, vol_ratio=2.0)
    _, breaks, states = _scan_events(100.0, 101.0, *args)
    assert len(breaks) == 1 and breaks[0].clean is True
    assert any(s == "broken" for _, s in states)


def test_break_then_reclaim_then_break_again_loops():
    """The state machine must never wedge; break/reclaim can repeat forever."""
    # break (98) -> reclaim inside (100.5) -> re-engage -> break -> reclaim
    closes = [105, 105, 98, 100.5, 105, 105, 98, 100.5, 105]
    args = _arrays(closes, lows=closes, highs=closes, vol_ratio=2.0)
    _, breaks, states = _scan_events(100.0, 101.0, *args)

    assert len(breaks) == 2, f"expected two distinct breaks, got {len(breaks)}"
    assert all(b.reclaimed for b in breaks)
    zone = _zone(breaks=breaks, states=states)
    assert zone.break_cycles_as_of(len(closes) - 1) == 2


def test_price_far_below_does_not_re_break_every_bar():
    """A zone left overhead must break once, not once per bar for years."""
    closes = [105, 105] + [50] * 300
    args = _arrays(closes, lows=closes, highs=closes, vol_ratio=2.0)
    _, breaks, _ = _scan_events(100.0, 101.0, *args)
    assert len(breaks) == 1


# ── Zone construction ───────────────────────────────────────────────────


def _synthetic_bars(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 50.0
    rows = []
    for _ in range(n):
        price = max(5.0, price * (1 + rng.normal(0, 0.012)))
        rng_hl = abs(rng.normal(0, 0.008)) * price
        close = price
        rows.append({
            "open": close - rng.normal(0, 0.003) * price,
            "high": close + rng_hl, "low": close - rng_hl,
            "close": close, "volume": float(rng.integers(1_000_000, 5_000_000)),
        })
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(rows, index=idx)


def test_no_dollar_spaced_levels_are_emitted():
    """Regression guard: the $1 round-number ladder is gone for good.

    The old build_level_map emitted every whole dollar and, because clustering
    sorted by price and kept the first entry, deleted real pivots sitting just
    above one. If zone anchors cluster on integers again, that bug is back.
    """
    zones = build_zones(_synthetic_bars())
    anchors = [z.anchor for z in zones]
    if len(anchors) < 5:
        pytest.skip("not enough zones in the synthetic frame to assess")
    near_integer = sum(1 for a in anchors if abs(a - round(a)) < 0.02)
    assert near_integer < len(anchors) * 0.5


def test_build_zones_requires_ohlcv():
    bars = _synthetic_bars(120).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        build_zones(bars)


def test_build_zones_returns_nothing_on_short_history():
    assert build_zones(_synthetic_bars(30)) == []


def test_zone_bands_scale_with_price():
    """Bands are ATR-relative, so they mean the same thing at $20 and $200."""
    zones = build_zones(_synthetic_bars(500))
    if len(zones) < 3:
        pytest.skip("not enough zones")
    widths_pct = [(z.hi - z.lo) / z.anchor for z in zones]
    assert max(widths_pct) - min(widths_pct) < 0.01


# ── Volume profile ──────────────────────────────────────────────────────


def test_poc_lands_in_the_heaviest_price_bin():
    heavy = [{"high": 101, "low": 99, "volume": 1_000_000} for _ in range(30)]
    light = [{"high": 112, "low": 108, "volume": 20_000} for _ in range(10)]
    profile = volume_profile(pd.DataFrame(heavy + light), bins=40)
    assert 99.0 <= poc(profile) <= 101.0


def test_high_volume_nodes_ignore_thin_regions():
    """HVN is a fraction of POC volume, not a quantile over bins.

    A quantile threshold is dragged down by the many empty bins in a trending
    name until thin price regions qualify as 'high volume'.
    """
    heavy = [{"high": 101, "low": 99, "volume": 1_000_000} for _ in range(40)]
    thin = [{"high": 112, "low": 108, "volume": 50_000} for _ in range(20)]
    nodes = high_volume_nodes(volume_profile(pd.DataFrame(heavy + thin), bins=40))
    assert len(nodes) == 1
    assert 99.0 <= nodes[0] <= 101.0


def test_volume_is_conserved_including_zero_range_bars():
    frame = pd.DataFrame([
        {"high": 101, "low": 99, "volume": 1000},
        {"high": 100, "low": 100, "volume": 500},    # limit day, zero range
        {"high": 105, "low": 103, "volume": 800},
    ])
    assert volume_profile(frame, bins=20).sum() == pytest.approx(2300.0)


def test_volume_profile_rejects_missing_columns():
    with pytest.raises(ValueError):
        volume_profile(pd.DataFrame({"high": [1.0], "low": [0.5]}), bins=5)
