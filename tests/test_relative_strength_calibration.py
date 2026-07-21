"""Tests for the cross-sectional relative-strength calibration probe.

The seal test is the one that matters: the RS ranks that label an event are computed
from the t-1 cross-section, and the SPY pullback filter from bars <= t-1. If day-t (or
later) data could move either, the probe would rank stocks using knowledge of the
reversal it is trying to measure. The two resolution tests pin the gap-order and
collision rules the spec is explicit about.
"""
from __future__ import annotations

import numpy as np

from scripts.relative_strength_calibration import (
    _assign_deciles,
    _excess_returns,
    _resolve_1r,
    _spy_pullback_ok,
    _spy_reversal,
)


# ── The seal: t-1 ranks and the pullback flag cannot see day t or later ────


def test_rs_seal_does_not_leak():
    """Decile ranks and the SPY pullback flag at t-1 are invariant to day-t+ data."""
    n, m = 60, 40
    t = 40
    rng = np.random.default_rng(0)
    # A rising, well-separated cross-section so deciles are stable.
    base = np.linspace(1.0, 5.0, m)
    close_mat = np.cumprod(1 + rng.normal(0.0005, 0.01, size=(n, m)), axis=0) * base
    spy_close = np.cumprod(1 + rng.normal(0.0003, 0.008, size=n)) * 100.0
    spy_high = spy_close * 1.02
    spy_atr = np.full(n, 1.0)

    excess_full = _excess_returns(close_mat, spy_close, t)
    deciles_full = _assign_deciles(excess_full)
    pullback_full = _spy_pullback_ok(spy_high, spy_close, spy_atr, t)

    # Truncate strictly before day t: rows > t-1 removed entirely.
    excess_trunc = _excess_returns(close_mat[:t], spy_close[:t], t)
    deciles_trunc = _assign_deciles(excess_trunc)
    np.testing.assert_array_equal(
        np.nan_to_num(deciles_trunc, nan=-1), np.nan_to_num(deciles_full, nan=-1)
    )

    # Poison every value at t and beyond — t-1 ranks and pullback must not move.
    cm = close_mat.copy()
    cm[t:] = 9_999.0
    sc = spy_close.copy()
    sc[t:] = 9_999.0
    sh = spy_high.copy()
    sh[t:] = 9_999.0
    np.testing.assert_array_equal(
        np.nan_to_num(_assign_deciles(_excess_returns(cm, sc, t)), nan=-1),
        np.nan_to_num(deciles_full, nan=-1),
    )
    assert _spy_pullback_ok(sh, sc, spy_atr, t) == pullback_full


def test_excess_returns_rank_top_and_bottom_correctly():
    """The stock that outran SPY most lands in D10; the worst in D1."""
    n, m = 30, 20
    t = 25
    close_mat = np.ones((n, m))
    # Give each ticker a distinct 10-bar return; SPY flat.
    for i in range(m):
        close_mat[t - 1, i] = 1.0 + 0.01 * i          # P[t-1]
        close_mat[t - 1 - 10, i] = 1.0                 # P[t-11]
    spy_close = np.ones(n)
    dec = _assign_deciles(_excess_returns(close_mat, spy_close, t))
    assert dec[np.argmax(np.arange(m))] == 10          # best performer -> D10
    assert dec[0] == 1                                 # worst performer -> D1


def test_spy_pullback_and_reversal_boundaries():
    n = 30
    t = 20
    high = np.full(n, 100.0)
    close = np.full(n, 100.0)
    atr_arr = np.full(n, 2.0)
    # No drawdown -> no pullback.
    assert not _spy_pullback_ok(high, close, atr_arr, t)
    # A 10-bar high of 104 with close[t-1]=100 -> drop 4 >= 1.5*2=3 -> pullback.
    high[t - 5] = 104.0
    assert _spy_pullback_ok(high, close, atr_arr, t)

    open_ = np.full(n, 100.0)
    close2 = np.full(n, 100.0)
    high2 = np.full(n, 100.0)
    assert not _spy_reversal(open_, close2, high2, t)   # doji, not above prior high
    close2[t] = 101.0                                   # green close
    assert _spy_reversal(open_, close2, high2, t)


# ── Resolution mechanics: gap order and collision ─────────────────────────


def _series(n=50, entry=100.0, atr_val=2.0):
    open_ = np.full(n, entry)
    high = np.full(n, entry)
    low = np.full(n, entry)
    atr_arr = np.full(n, atr_val)
    return open_, high, low, atr_arr


def test_gap_resolution_order():
    """A gap-down open past the stop is a loss even if that bar's high tags target."""
    t = 20
    o, h, low, atr_arr = _series()          # entry=open[t+1]=100, target=102, stop=98
    j = t + 3
    o[j] = 97.0                              # gaps below the 98 stop
    h[j] = 103.0                             # ...and later trades above the 102 target
    low[j] = 97.0
    outcome, _ = _resolve_1r(o, h, low, atr_arr, t)
    assert outcome == "loss"


def test_gap_up_open_is_an_immediate_win():
    t = 20
    o, h, low, atr_arr = _series()
    j = t + 2
    o[j] = 102.5                             # gaps above the 102 target
    h[j] = 103.0
    low[j] = 102.0
    outcome, bars = _resolve_1r(o, h, low, atr_arr, t)
    assert outcome == "win"
    assert bars == float(j - t)


def test_same_bar_collision_is_loss():
    """A non-gap bar spanning both target and stop scores conservatively as a loss."""
    t = 20
    o, h, low, atr_arr = _series()
    j = t + 4
    o[j] = 100.0                             # opens inside the band (no gap)
    h[j] = 102.5                             # high tags target
    low[j] = 97.5                            # low tags stop, same bar
    outcome, _ = _resolve_1r(o, h, low, atr_arr, t)
    assert outcome == "loss"
