"""Tests for the TA-confluence calibration probe.

The seal test is the one that matters: the confluence score that labels a touch is
computed from conditions read at t-1. If any day-t (or later) data could change the
score, the probe would be scoring the entry with knowledge of its own outcome. The
mechanics (entry = open[t+1], risk from ATR[t-1]) are inherited from `_resolve_bounce`
and already covered in `test_trend_pullback_calibration`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.confluence_calibration import (
    N_CONDITIONS,
    RSI_MAX,
    STRETCH_MAX,
    TREND_MATURITY_BARS,
    _confluence_score,
)
from scripts.trend_pullback_calibration import _regime_run, trigger_lines
from signals.indicators.momentum import macd, rsi


def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2005-01-01", periods=len(closes), freq="B")
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


# ── The seal: every condition read at t-1 is invariant to day-t data ──────


def test_confluence_seal_does_not_leak():
    """The confluence score labelling a touch cannot see bar t or later.

    Each ingredient (regime run, RSI, MACD histogram) evaluated at t-1 must be
    bit-identical whether or not any bar >= t exists — and poisoning day t onward
    must not move it. Zones (C4) are causal by construction (built from a window
    strictly before t) and are exercised in the full run.
    """
    t = 260
    closes = 100.0 + np.linspace(0.0, 60.0, 320)     # a long, gentle uptrend

    ema20_full, ema50_full, ema200_full, _ = trigger_lines(_ohlcv(closes))
    run_full = _regime_run(ema20_full, ema50_full, ema200_full)
    rsi_full = rsi(_ohlcv(closes)["close"]).to_numpy()
    hist_full = macd(_ohlcv(closes)["close"]).histogram.to_numpy()

    # Truncate strictly before day t and recompute from scratch.
    trunc = _ohlcv(closes[:t])
    e20, e50, e200, _ = trigger_lines(trunc)
    run_t = _regime_run(e20, e50, e200)
    rsi_t = rsi(trunc["close"]).to_numpy()
    hist_t = macd(trunc["close"]).histogram.to_numpy()

    assert run_t[-1] == run_full[t - 1]
    assert rsi_t[-1] == rsi_full[t - 1]
    assert hist_t[-1] == hist_full[t - 1]

    # Poisoning day t onward cannot move any t-1 value.
    poisoned = closes.copy()
    poisoned[t:] = 10_000.0
    e20p, e50p, e200p, _ = trigger_lines(_ohlcv(poisoned))
    assert _regime_run(e20p, e50p, e200p)[t - 1] == run_full[t - 1]
    assert rsi(_ohlcv(poisoned)["close"]).to_numpy()[t - 1] == rsi_full[t - 1]
    assert macd(_ohlcv(poisoned)["close"]).histogram.to_numpy()[t - 1] == hist_full[t - 1]


# ── Scoring: count only sealed conditions, at their documented thresholds ──


def _arrays(n=50):
    ema20 = np.full(n, 100.0)
    atr_arr = np.full(n, 2.0)
    high = np.full(n, 101.0)
    run = np.full(n, TREND_MATURITY_BARS + 5)        # C1 satisfied
    rsi_arr = np.full(n, RSI_MAX - 5.0)              # C2 satisfied
    hist_arr = np.full(n, 1.0)                       # C3 satisfied
    return ema20, atr_arr, high, run, rsi_arr, hist_arr


def test_all_five_conditions_can_score_five():
    t = 30
    ema20, atr_arr, high, run, rsi_arr, hist_arr = _arrays()
    # C5: a flat leg keeps stretch at (101-100)/2 = 0.5 <= STRETCH_MAX.

    class _Zone:
        lo, hi = 99.0, 101.0
        def strength_as_of(self, _):
            return 0.9                                # C4 satisfied

    score = _confluence_score(t, ema20, atr_arr, high, run, rsi_arr, hist_arr,
                              [_Zone()], z_last=t - 1, leg_start=0)
    assert score == N_CONDITIONS


def test_conditions_respect_their_thresholds():
    t = 30
    ema20, atr_arr, high, run, rsi_arr, hist_arr = _arrays()

    # No zones -> C4 fails; everything else holds -> 4.
    assert _confluence_score(t, ema20, atr_arr, high, run, rsi_arr, hist_arr,
                             [], z_last=t - 1, leg_start=0) == N_CONDITIONS - 1

    # Trip C1 (immature trend) and C2 (RSI too hot): score drops to 2.
    run2 = run.copy()
    run2[t - 1] = TREND_MATURITY_BARS - 1
    rsi2 = rsi_arr.copy()
    rsi2[t - 1] = RSI_MAX + 5.0
    assert _confluence_score(t, ema20, atr_arr, high, run2, rsi2, hist_arr,
                             [], z_last=t - 1, leg_start=0) == N_CONDITIONS - 3


def test_c5_fails_when_the_leg_is_overstretched():
    t = 30
    ema20, atr_arr, high, run, rsi_arr, hist_arr = _arrays()
    high = high.copy()
    high[15] = 100.0 + (STRETCH_MAX + 2.0) * 2.0      # a >STRETCH_MAX spike in-leg
    # No zones (C4 off); C1-C3 on, C5 now off -> 3.
    score = _confluence_score(t, ema20, atr_arr, high, run, rsi_arr, hist_arr,
                              [], z_last=t - 1, leg_start=0)
    assert score == N_CONDITIONS - 2


def test_score_reads_t_minus_one_not_t():
    """Poisoning the touch bar's own condition inputs must not change the score."""
    t = 30
    ema20, atr_arr, high, run, rsi_arr, hist_arr = _arrays()
    baseline = _confluence_score(t, ema20, atr_arr, high, run, rsi_arr, hist_arr,
                                 [], z_last=t - 1, leg_start=0)
    run[t] = 0        # wreck bar t only — must not affect a t-1 score
    rsi_arr[t] = 99.0
    hist_arr[t] = -5.0
    poisoned = _confluence_score(t, ema20, atr_arr, high, run, rsi_arr, hist_arr,
                                 [], z_last=t - 1, leg_start=0)
    assert poisoned == baseline
