"""Tests for the breakdown calibration probe.

The seal test is the one that matters. The strength that bins a break is computed
before the break exists; if any post-break information could change it, the study
would be predicting the flush from the flush. The mechanics tests pin the two
other places a leak hides: the entry must be the open *after* the break, and the
risk must be sized from ATR at b-1, never from the break bar's own (large) range.
"""
from __future__ import annotations

import numpy as np

from scripts.breakdown_calibration import (
    BREAK_CLOSE_ATR,
    STOP_ATR,
    _first_break,
    _resolve_break,
)
from signals.price_action.level_quality import REACTION_WINDOW, Touch, ZoneTrack


def _zone(touches=()) -> ZoneTrack:
    return ZoneTrack(
        lo=100.0, hi=101.0, anchor=100.5, origins=frozenset({"pivot"}),
        touches=tuple(touches), breaks=(), volume_share=0.5, _states=(),
    )


def _touch(bar_idx: int, reaction: float = 2.0) -> Touch:
    return Touch(bar_idx=bar_idx, penetration_atr=0.2, reaction_atr=reaction, volume_ratio=1.0)


# ── The seal: strength binning a break cannot see past the break ──────────


def test_strength_at_a_break_does_not_leak_forward():
    """The strength decile that labels a break is sealed strictly before it.

    Build a zone whose sealed strength is fixed as of the break bar, then let the
    future arrive — including huge new touches. The strength that would bin the
    break must be bit-identical. If it moves, the calibration is scoring breaks
    with knowledge of what happened after them.
    """
    break_bar = 60
    # Touches whose 10-bar reaction windows all close before the break bar.
    history = [_touch(0), _touch(20), _touch(40)]
    sealed_before = _zone(history).strength_as_of(break_bar)

    # The future turns out dramatic: fresh, high-reaction touches after the break.
    with_future = _zone(history + [_touch(65, reaction=9.0), _touch(80, reaction=9.0)])
    assert with_future.strength_as_of(break_bar) == sealed_before

    # And a touch still inside its reaction window at the break contributes nothing.
    unresolved = _touch(break_bar - 3)
    assert unresolved.sealed_at > break_bar - 1
    assert _zone(history + [unresolved]).strength_as_of(break_bar) == sealed_before


def test_touch_reaction_window_must_close_before_it_counts():
    resolved = _touch(30 - REACTION_WINDOW - 1)
    assert resolved.sealed_at <= 30 - 1
    assert _zone([resolved]).strength_as_of(30) > 0.0


# ── Mechanics: entry, stop, target, outcomes ─────────────────────────────


def _frame(n: int = 80, atr_val: float = 2.0):
    """Flat scaffold: price sits above the lo=100/hi=101 zone doing nothing."""
    open_ = np.full(n, 105.0)
    high = np.full(n, 106.0)
    low = np.full(n, 104.0)
    close = np.full(n, 105.0)
    atr_arr = np.full(n, atr_val)
    vol_avg = np.full(n, 1_000_000.0)
    volume = np.full(n, 1_000_000.0)
    return open_, high, low, close, atr_arr, vol_avg, volume


def test_flush_uses_next_bar_open_as_entry():
    """entry = open[b+1]; a 1R flush is measured from there, not the break close."""
    b = 40

    def frame_with_post_break_at(fill: float):
        o, h, low, c, atr_arr, _, _ = _frame()
        c[b] = 98.0                   # closes below lo - 0.5*atr = 99 -> a break
        for j in range(b + 1, len(c)):
            o[j] = h[j] = low[j] = c[j] = fill   # drift below the zone, clear of the stop
        return o, h, low, c, atr_arr

    # Entry = open[b+1] = 98. stop = hi + 0.2*atr = 101.4 ; risk = 3.4 ; target = 94.6.
    o, h, low, c, atr_arr = frame_with_post_break_at(98.0)
    low[b + 3] = 94.0                   # trades through the 1R target
    outcome, bars = _resolve_break(o, h, low, c, atr_arr, b, 100.0, 101.0)
    assert outcome == "flush"
    assert bars == 3.0                # b+3 minus b

    # Raise the entry: open[b+1] = 100.5 -> risk = 0.9, target = 99.6. The very
    # same 94.0 dip would still flush, so keep the floor above 99.6 to isolate the
    # entry: nothing reaches the higher target, proving open[b+1] drove the result.
    o, h, low, c, atr_arr = frame_with_post_break_at(100.0)
    o[b + 1] = 100.5
    low[b + 3] = 99.7                   # above the 99.6 target
    outcome2, _ = _resolve_break(o, h, low, c, atr_arr, b, 100.0, 101.0)
    assert outcome2 != "flush"


def test_risk_is_sized_from_atr_at_b_minus_1_not_the_break_bar():
    """The break bar's own range must not inflate the stop distance."""
    o, h, low, c, atr_arr, _, _ = _frame()
    b = 40
    c[b] = 98.0
    for j in range(b + 1, len(c)):
        o[j] = h[j] = low[j] = c[j] = 98.0
    o[b + 1] = 98.0
    low[b + 3] = 94.0
    baseline, _ = _resolve_break(o, h, low, c, atr_arr, b, 100.0, 101.0)

    # Poison ONLY the break bar's ATR with an absurd value. If the resolver reads
    # atr[b] the stop/target move and the outcome changes; sealed at b-1 it can't.
    atr_arr[b] = 999.0
    poisoned, _ = _resolve_break(o, h, low, c, atr_arr, b, 100.0, 101.0)
    assert poisoned == baseline == "flush"


def test_bear_trap_reclaim_is_a_sweep_not_a_flush():
    o, h, low, c, atr_arr, _, _ = _frame()
    b = 40
    c[b] = 98.0
    for j in range(b + 1, len(c)):
        o[j] = h[j] = low[j] = c[j] = 98.0    # below the zone, clear of the stop
    o[b + 1] = 98.0
    # stop = 101.4 ; a close back above it within SWEEP_WINDOW, before any flush
    c[b + 2] = 102.0
    outcome, bars = _resolve_break(o, h, low, c, atr_arr, b, 100.0, 101.0)
    assert outcome == "sweep"
    assert bars is None


def test_grind_that_hits_neither_target_nor_stop_is_neither():
    o, h, low, c, atr_arr, _, _ = _frame()
    b = 40
    c[b] = 98.0
    o[b + 1] = 98.0
    # Drift sideways just below the zone: never reaches 94.6 target nor 101.4 stop.
    for j in range(b + 1, min(b + 1 + 40, len(c))):
        o[j] = h[j] = low[j] = c[j] = 97.0
    outcome, _ = _resolve_break(o, h, low, c, atr_arr, b, 100.0, 101.0)
    assert outcome == "neither"


# ── _first_break: engagement + volume gate ───────────────────────────────


def test_first_break_needs_real_volume():
    o, h, low, c, atr_arr, vol_avg, volume = _frame()
    b = 40
    c[b] = 98.0                       # below the 99 break threshold
    volume[b] = 900_000.0             # only 0.9x average -> not a clean break
    assert _first_break(c, volume, atr_arr, vol_avg, 30, 30, 100.0, 101.0) is None

    volume[b] = 2_000_000.0           # 2x average -> clean
    assert _first_break(c, volume, atr_arr, vol_avg, 30, 30, 100.0, 101.0) == b


def test_first_break_requires_prior_engagement():
    """A level already far below price cannot register a fresh break."""
    o, h, low, c, atr_arr, vol_avg, volume = _frame()
    c[:] = 50.0                       # price sits far below the 100/101 zone
    volume[:] = 2_000_000.0
    assert _first_break(c, volume, atr_arr, vol_avg, 30, 30, 100.0, 101.0) is None


def test_break_threshold_matches_the_documented_rule():
    """Regression guard on the constants the trigger is defined by."""
    o, h, low, c, atr_arr, vol_avg, volume = _frame(atr_val=2.0)
    b = 40
    volume[b] = 2_000_000.0
    c[b] = 100.0 - BREAK_CLOSE_ATR * 2.0 + 0.01   # just above threshold -> no break
    assert _first_break(c, volume, atr_arr, vol_avg, 30, 30, 100.0, 101.0) is None
    c[b] = 100.0 - BREAK_CLOSE_ATR * 2.0 - 0.01   # just below -> break
    assert _first_break(c, volume, atr_arr, vol_avg, 30, 30, 100.0, 101.0) == b


def test_stop_constant_is_wired():
    """A close exactly at hi + STOP_ATR*ATR is the failure boundary."""
    o, h, low, c, atr_arr, _, _ = _frame()
    b = 40
    c[b] = 98.0
    for j in range(b + 1, len(c)):
        o[j] = h[j] = low[j] = c[j] = 98.0
    o[b + 1] = 98.0
    stop = 101.0 + STOP_ATR * 2.0
    c[b + 2] = stop + 0.5              # clearly above the stop -> sweep
    outcome, _ = _resolve_break(o, h, low, c, atr_arr, b, 100.0, 101.0)
    assert outcome == "sweep"
