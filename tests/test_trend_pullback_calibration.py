"""Tests for the trend-pullback (dynamic mean-reversion) calibration probe.

The seal test is the one that matters. The trigger line for day t is the 20-EMA as
of t-1; if day-t (or later) data could move that line, the probe would be predicting
the touch from the touch. `test_dynamic_seal_does_not_leak` truncates the frame at
t-1 and proves the line is bit-identical. The mechanics tests pin the other places a
leak hides: the entry must be the open *after* the touch, and the risk must be sized
from ATR at t-1, never the touch bar's own range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.trend_pullback_calibration import (
    REGIME_BARS,
    _is_touch,
    _leg_max_stretch,
    _regime_run,
    _resolve_bounce,
    trigger_lines,
)


# ── The seal: the day-t trigger line is the 20-EMA as of t-1 ──────────────


def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000.0),
        },
        index=idx,
    )


def test_dynamic_seal_does_not_leak():
    """The 20-EMA line evaluated on day t uses only closes <= t-1.

    Recomputing the EMA on the frame truncated at t-1 must reproduce, bit-for-bit,
    the value the full series reports at t-1 — and mutating day t onward to extreme
    values must not move it. If it moves, the touch on day t is being tested against
    a line that already knows day t happened.
    """
    t = 50
    closes = 100.0 + np.linspace(0.0, 20.0, 60)  # a gently rising trend

    ema20_full = trigger_lines(_ohlcv(closes))[0]
    line_for_day_t = ema20_full[t - 1]

    # Truncate strictly before day t and recompute from scratch.
    ema20_trunc = trigger_lines(_ohlcv(closes[:t]))[0]
    assert ema20_trunc[-1] == line_for_day_t          # index t-1 is the last row

    # The future turns out extreme: nothing about day t.. can move the t-1 line.
    poisoned = closes.copy()
    poisoned[t:] = 10_000.0
    ema20_poisoned = trigger_lines(_ohlcv(poisoned))[0]
    assert ema20_poisoned[t - 1] == line_for_day_t


def test_touch_is_evaluated_against_the_t_minus_1_line():
    ema20 = np.full(10, 100.0)
    low = np.full(10, 105.0)
    high = np.full(10, 106.0)
    # Range sits above the mean -> not a touch.
    assert not _is_touch(low, high, ema20, 5)
    # Drop the low through the (t-1) line -> a touch.
    low[5] = 99.0
    assert _is_touch(low, high, ema20, 5)


# ── Mechanics: entry, symmetric 1R, ATR sealing ──────────────────────────


def _bars(n: int = 80, atr_val: float = 2.0):
    """Flat scaffold: price sits at 100, ATR is 2.0 everywhere."""
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    atr_arr = np.full(n, atr_val)
    return open_, high, low, atr_arr


def test_entry_uses_next_bar_open():
    """entry = open[t+1]; the 1R is measured from there, not the touch close."""
    t = 40
    o, h, low, atr_arr = _bars()          # entry=100, target=102, stop=98
    h[t + 3] = 102.5                       # trades through the +1R target
    outcome, bars = _resolve_bounce(o, h, low, atr_arr, t)
    assert outcome == "win"
    assert bars == 3.0

    # Raise the entry to 101 -> target=103. The same 102.5 high no longer reaches it,
    # and the low never hits the 99 stop, so the outcome flips to neither. Only the
    # entry (open[t+1]) changed — proving it drove the result.
    o, h, low, atr_arr = _bars()
    o[t + 1] = 101.0
    h[t + 3] = 102.5
    outcome2, _ = _resolve_bounce(o, h, low, atr_arr, t)
    assert outcome2 != "win"


def test_risk_is_sized_from_atr_at_t_minus_1_not_the_touch_bar():
    """The touch bar's own ATR must not resize the target/stop."""
    t = 40
    o, h, low, atr_arr = _bars()
    h[t + 3] = 102.5
    baseline, _ = _resolve_bounce(o, h, low, atr_arr, t)

    atr_arr[t] = 999.0                     # poison ONLY the touch bar's ATR
    poisoned, _ = _resolve_bounce(o, h, low, atr_arr, t)
    assert poisoned == baseline == "win"


def test_target_and_stop_are_symmetric_one_R():
    """Target = entry + ATR(t-1); stop = entry - ATR(t-1); a same-bar tie is a loss."""
    t = 40
    # Target-only hit -> win.
    o, h, low, atr_arr = _bars()
    h[t + 2] = 102.0                       # exactly +1R
    assert _resolve_bounce(o, h, low, atr_arr, t)[0] == "win"

    # Stop-only hit -> loss (and it is exactly 1 ATR away, mirroring the target).
    o, h, low, atr_arr = _bars()
    low[t + 2] = 98.0                      # exactly -1R
    assert _resolve_bounce(o, h, low, atr_arr, t)[0] == "loss"

    # Same bar spans both -> conservative loss (stop assumed first).
    o, h, low, atr_arr = _bars()
    h[t + 2] = 102.0
    low[t + 2] = 98.0
    assert _resolve_bounce(o, h, low, atr_arr, t)[0] == "loss"


# ── Regime: the strict stack must hold for REGIME_BARS consecutive bars ────


def test_regime_requires_ten_consecutive_bars():
    n = 40
    ema20 = np.full(n, 3.0)
    ema50 = np.full(n, 2.0)
    ema200 = np.full(n, 1.0)               # stack 20>50>200 holds everywhere...
    ema20[15] = 0.0                        # ...except one break at bar 15
    run = _regime_run(ema20, ema50, ema200)

    assert run[14] == 15                   # 0..14 held
    assert run[15] == 0                    # break resets the counter
    assert run[25] == 10                   # 16..25 is exactly 10 consecutive

    # The collect rule is run[t-1] >= REGIME_BARS: a trigger at t=26 just clears it,
    # a trigger at t=25 (run[24]=9) does not.
    assert run[25] >= REGIME_BARS
    assert run[24] < REGIME_BARS


# ── Stretch: max excursion measured only over the leg, ending at t-1 ───────


def test_stretch_measured_since_last_touch():
    n = 30
    ema20 = np.full(n, 100.0)
    atr_arr = np.full(n, 2.0)
    high = np.full(n, 101.0)               # baseline stretch 0.5 ATR everywhere
    high[10] = 106.0                        # a 3.0-ATR spike inside the leg
    high[3] = 120.0                         # a huge spike BEFORE the leg starts
    high[20] = 130.0                        # a huge spike AT/AFTER the touch bar

    # Leg spans (last touch = 5, current touch t = 20): bars [6, 20).
    stretch = _leg_max_stretch(high, ema20, atr_arr, start=6, end=20)
    assert stretch == (106.0 - 100.0) / 2.0        # the in-leg spike, = 3.0

    # Pre-leg (bar 3) and the touch bar itself (bar 20) are excluded -> no leakage.
    assert stretch < (120.0 - 100.0) / 2.0
    assert stretch < (130.0 - 100.0) / 2.0
