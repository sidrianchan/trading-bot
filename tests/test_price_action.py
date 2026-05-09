"""Unit tests for signals/price_action/*."""
from __future__ import annotations

import pandas as pd
import pytest

from signals.price_action.support_resistance import (
    Level, LevelMap,
    find_swing_pivots, round_number_levels, prior_session_levels, build_level_map,
)
from signals.price_action.candlesticks import (
    hammer, bullish_engulfing, morning_star, bullish_pin_bar,
    shooting_star, bearish_engulfing, evening_star, bearish_pin_bar,
    detect_all,
)
from signals.price_action.breakouts import consolidation_breakout, flag_breakout


def _bar(o, h, l, c, v=1_000):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── Support / resistance ───────────────────────────────────────────────────


class TestSwingPivots:
    def test_finds_central_high(self):
        # peak at index 5
        bars = _df([
            _bar(100, 101, 99, 100),
            _bar(100, 102, 99, 101),
            _bar(101, 103, 100, 102),
            _bar(102, 104, 101, 103),
            _bar(103, 105, 102, 104),
            _bar(104, 110, 103, 109),  # ← swing high
            _bar(108, 109, 105, 106),
            _bar(106, 107, 104, 105),
            _bar(105, 106, 103, 104),
            _bar(104, 105, 102, 103),
            _bar(103, 104, 101, 102),
        ])
        highs, lows = find_swing_pivots(bars, window=5)
        assert 110.0 in highs

    def test_no_pivots_when_too_few_bars(self):
        bars = _df([_bar(100, 101, 99, 100) for _ in range(5)])
        assert find_swing_pivots(bars, window=5) == ([], [])


class TestRoundNumbers:
    def test_generates_in_range(self):
        out = round_number_levels(98.0, 153.0, ladders=[5, 50])
        assert 100.0 in out and 150.0 in out and 105.0 in out
        assert all(98.0 <= v <= 153.0 for v in out)

    def test_empty_when_range_inverted(self):
        assert round_number_levels(150.0, 100.0) == []


class TestPriorSessionLevels:
    def test_pdh_pdl_extracted(self):
        idx = pd.date_range("2024-01-08", periods=6, freq="B")  # Mon..Mon
        df = pd.DataFrame({
            "open":  [100, 101, 102, 103, 104, 105],
            "high":  [101, 103, 102, 104, 106, 107],
            "low":   [ 99, 100, 101, 102, 103, 104],
            "close": [100, 102, 102, 103, 105, 106],
            "volume": [1] * 6,
        }, index=idx)
        out = prior_session_levels(df)
        assert out["pdh"] == 106.0
        assert out["pdl"] == 103.0

    def test_pwh_pwl_extracted_when_history_spans_two_weeks(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "open":  list(range(100, 110)),
            "high":  list(range(101, 111)),
            "low":   list(range(99, 109)),
            "close": list(range(100, 110)),
            "volume": [1] * 10,
        }, index=idx)
        out = prior_session_levels(df)
        # PWH/PWL should be present (prior calendar week of business days)
        assert "pwh" in out and "pwl" in out


class TestLevelMap:
    def test_role_assignment(self):
        bars = _df([_bar(100 + i, 101 + i, 99 + i, 100 + i) for i in range(40)])
        lm = build_level_map(bars, swing_window=3, round_ladders=[10])
        last = bars["close"].iloc[-1]
        for lvl in lm.levels:
            if lvl.price > last:
                assert lvl.role == "resistance"
            elif lvl.price < last:
                assert lvl.role == "support"

    def test_near_finds_close_levels(self):
        lm = LevelMap(
            levels=[Level(100.0, "round", "resistance"), Level(110.0, "round", "resistance")],
            last_price=99.0,
        )
        nearby = lm.near(100.4, tolerance_pct=0.005)
        assert any(l.price == 100.0 for l in nearby)
        assert all(l.price != 110.0 for l in nearby)


# ── Candlesticks ───────────────────────────────────────────────────────────


class TestCandlesticks:
    def test_hammer_triggered(self):
        # small body at top, long lower wick
        bars = _df([_bar(o=101.0, h=101.5, l=95.0, c=101.2)])
        assert hammer(bars).triggered

    def test_hammer_skips_normal_bar(self):
        bars = _df([_bar(100, 101, 99, 100.8)])
        assert not hammer(bars).triggered

    def test_bullish_engulfing(self):
        bars = _df([_bar(102, 103, 100, 100.5), _bar(100.0, 104, 99.5, 103.5)])
        assert bullish_engulfing(bars).triggered

    def test_bullish_engulfing_requires_prior_red(self):
        bars = _df([_bar(100, 104, 100, 103), _bar(102, 105, 100, 104)])
        assert not bullish_engulfing(bars).triggered

    def test_morning_star(self):
        bars = _df([
            _bar(110, 110, 100, 101),     # long red
            _bar(100, 101, 99, 100.3),    # small body
            _bar(100, 110, 99, 109),      # long green into bar1 body
        ])
        assert morning_star(bars).triggered

    def test_bullish_pin_bar(self):
        bars = _df([_bar(o=100.6, h=101.0, l=95.0, c=101.0)])
        assert bullish_pin_bar(bars).triggered

    def test_shooting_star(self):
        bars = _df([_bar(o=100, h=110, l=99.5, c=100.3)])
        assert shooting_star(bars).triggered

    def test_bearish_engulfing(self):
        bars = _df([_bar(100, 102, 99.5, 102), _bar(102.5, 103, 99, 99.5)])
        assert bearish_engulfing(bars).triggered

    def test_evening_star(self):
        bars = _df([
            _bar(100, 110, 99, 109),      # long green
            _bar(109, 110, 108.5, 109.2), # small body
            _bar(109, 109.5, 99, 100),    # long red into bar1 body
        ])
        assert evening_star(bars).triggered

    def test_bearish_pin_bar(self):
        bars = _df([_bar(o=99, h=110, l=98.5, c=99.0)])
        assert bearish_pin_bar(bars).triggered

    def test_detect_all_returns_bullish_for_hammer(self):
        bars = _df([
            _bar(100, 100.5, 99, 100),
            _bar(o=101.0, h=101.5, l=95.0, c=101.2),
        ])
        names = [p.name for p in detect_all(bars)]
        assert "hammer" in names


# ── Breakouts ──────────────────────────────────────────────────────────────


class TestBreakouts:
    def test_consolidation_long_breakout(self):
        # First 36 wide bars (range ~1.0) then 24 tight bars (range ~0.2)
        # → ATR(14) on tight tail much smaller than ATR(50) over full window
        rows = [_bar(100, 100.5, 99.5, 100, v=1_000) for _ in range(36)]
        rows += [_bar(100, 100.1, 99.9, 100, v=1_000) for _ in range(24)]
        # latest bar: close above the prior 10-bar range high (~100.1) on 3× volume
        rows.append(_bar(100.05, 102, 100.0, 101.5, v=3_000))
        bars = _df(rows)
        sig = consolidation_breakout(bars, consolidation_window=10, atr_short=14, atr_long=50)
        assert sig.triggered, sig.detail
        assert sig.direction == "long"

    def test_consolidation_skips_without_volume(self):
        rows = [_bar(100, 100.5, 99.5, 100, v=1_000) for _ in range(36)]
        rows += [_bar(100, 100.1, 99.9, 100, v=1_000) for _ in range(24)]
        rows.append(_bar(100.05, 102, 100, 101.5, v=900))   # below avg volume
        bars = _df(rows)
        assert not consolidation_breakout(bars, consolidation_window=10).triggered

    def test_consolidation_skips_without_atr_contraction(self):
        # Wide range bars throughout — ATR(14)/ATR(50) won't contract
        rng = []
        for i in range(60):
            rng.append(_bar(100, 110, 90, 100 + (i % 2)))
        rng.append(_bar(100, 115, 99, 114, v=3_000))
        bars = _df(rng)
        sig = consolidation_breakout(bars, consolidation_window=10)
        assert not sig.triggered

    def test_flag_long_breakout(self):
        # Impulse: 100 -> 110 (5 bars), flag: oscillates near 109 (5 bars), break to 112
        impulse = [_bar(100 + i*2, 102 + i*2, 99 + i*2, 100 + i*2 + 1, v=2_000) for i in range(5)]
        flag = [_bar(109, 110, 108, 109, v=1_500) for _ in range(5)]
        last = _bar(109.5, 113, 109, 112, v=2_500)
        # Pad with 20 bars of prior history so volume_ratio(period=20) is meaningful
        pad = [_bar(99, 99.5, 98.5, 99, v=1_500) for _ in range(20)]
        bars = _df(pad + impulse + flag + [last])
        sig = flag_breakout(bars, impulse_window=5, flag_window=5, impulse_min_pct=0.05)
        assert sig.triggered
        assert sig.direction == "long"

    def test_flag_short_breakout(self):
        pad = [_bar(101, 101.5, 100.5, 101, v=1_500) for _ in range(20)]
        impulse = [_bar(100 - i*2, 100 - i*2, 98 - i*2, 99 - i*2, v=2_000) for i in range(5)]
        flag = [_bar(91, 92, 90, 91, v=1_500) for _ in range(5)]
        last = _bar(91, 91.5, 88, 89, v=2_500)
        bars = _df(pad + impulse + flag + [last])
        sig = flag_breakout(bars, impulse_window=5, flag_window=5, impulse_min_pct=0.05)
        assert sig.triggered
        assert sig.direction == "short"

    def test_flag_skips_without_impulse(self):
        flat = [_bar(100, 100.5, 99.5, 100, v=1_500) for _ in range(35)]
        last = _bar(100, 102, 100, 101.5, v=2_500)
        bars = _df(flat + [last])
        sig = flag_breakout(bars, impulse_window=5, flag_window=5, impulse_min_pct=0.05)
        assert not sig.triggered
