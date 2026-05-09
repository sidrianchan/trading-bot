"""Tests for intraday signal, risk, and position management logic."""
from __future__ import annotations

from datetime import datetime, date
from unittest.mock import MagicMock, patch

import pytest

from signals.gap import GapSignal, StockSnapshot
from signals.orb import ORBSignal
from signals.vwap import VWAPSignal
from signals.intraday_composite import IntradayComposite
from execution.orders import BracketOrder, Side
from portfolio.position_manager import PositionManager
from risk.limits import RiskLimits
from risk.drawdown import DrawdownMonitor


# ── Helpers ────────────────────────────────────────────────────────────────

def make_snapshot(ticker, prev_close=100.0, gap_pct=0.02, volume_ratio=3.0, atr=1.5):
    latest = prev_close * (1 + gap_pct)
    return StockSnapshot(
        ticker=ticker,
        prev_close=prev_close,
        latest_price=latest,
        pre_market_volume=int(500_000 * volume_ratio),
        avg_volume_30d=500_000.0,
        atr=atr,
        gap_pct=gap_pct,
        volume_ratio=volume_ratio,
    )


def make_bar(open_=100, high=101, low=99, close=100.5, volume=50_000):
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def make_orb_bars(n, base=100.0, direction="up"):
    bars = []
    for i in range(n):
        if direction == "up":
            bars.append(make_bar(base, base + 1.5, base - 0.5, base + 0.8))
        else:
            bars.append(make_bar(base, base + 0.5, base - 1.5, base - 0.8))
    return bars


# ── GapSignal ──────────────────────────────────────────────────────────────

class TestGapSignal:
    def test_no_gap_scores_zero(self):
        sig = GapSignal(gap_min_pct=1.5)
        snaps = [make_snapshot("AAPL", gap_pct=0.005)]
        scores = sig.score(snaps)
        assert scores.get("AAPL", 0) == 0

    def test_gap_up_scores_positive(self):
        sig = GapSignal(gap_min_pct=1.5)
        snaps = [make_snapshot("AAPL", gap_pct=0.025, volume_ratio=3.0)]
        scores = sig.score(snaps)
        assert scores.get("AAPL", 0) > 0

    def test_gap_down_scores_negative(self):
        sig = GapSignal(gap_min_pct=1.5)
        snaps = [make_snapshot("NVDA", gap_pct=-0.025, volume_ratio=3.0)]
        scores = sig.score(snaps)
        assert scores.get("NVDA", 0) < 0

    def test_low_volume_ratio_excluded(self):
        sig = GapSignal(gap_min_pct=1.5, volume_ratio_min=2.0)
        snaps = [make_snapshot("TSLA", gap_pct=0.03, volume_ratio=1.2)]
        scores = sig.score(snaps)
        assert scores.get("TSLA", 0) == 0

    def test_low_atr_excluded(self):
        sig = GapSignal(atr_min_dollars=0.50)
        snaps = [make_snapshot("XYZ", gap_pct=0.02, volume_ratio=3.0, atr=0.20)]
        scores = sig.score(snaps)
        assert scores.get("XYZ", 0) == 0

    def test_larger_gap_scores_higher(self):
        sig = GapSignal(gap_min_pct=1.5)
        small = make_snapshot("A", gap_pct=0.02, volume_ratio=3.0)
        large = make_snapshot("B", gap_pct=0.05, volume_ratio=3.0)
        scores = sig.score([small, large])
        assert scores["B"] > scores["A"]


# ── ORBSignal ──────────────────────────────────────────────────────────────

class TestORBSignal:
    def test_no_signal_during_orb_window(self):
        sig = ORBSignal(orb_minutes=15)
        for bar in make_orb_bars(10):
            result = sig.on_bar("AAPL", bar)
        assert result is None

    def test_range_established_after_orb_window(self):
        sig = ORBSignal(orb_minutes=5)
        for bar in make_orb_bars(5):
            sig.on_bar("AAPL", bar)
        assert sig.get_range("AAPL") is not None

    def test_breakout_long_detected(self):
        sig = ORBSignal(orb_minutes=3, volume_confirm_ratio=1.0)
        for bar in make_orb_bars(3, base=100.0):
            sig.on_bar("AAPL", bar)
        # Bar breaking above range high with high volume
        breakout_bar = make_bar(open_=101.5, high=103, low=101, close=102.5, volume=200_000)
        result = sig.on_bar("AAPL", breakout_bar)
        assert result == "long"

    def test_breakout_short_detected(self):
        sig = ORBSignal(orb_minutes=3, volume_confirm_ratio=1.0)
        for bar in make_orb_bars(3, base=100.0):
            sig.on_bar("AAPL", bar)
        # Bar breaking below range low
        breakout_bar = make_bar(open_=98.5, high=99, low=96.5, close=97.0, volume=200_000)
        result = sig.on_bar("AAPL", breakout_bar)
        assert result == "short"

    def test_reset_clears_state(self):
        sig = ORBSignal(orb_minutes=3)
        for bar in make_orb_bars(3):
            sig.on_bar("AAPL", bar)
        sig.reset()
        assert sig.get_range("AAPL") is None


# ── VWAPSignal ─────────────────────────────────────────────────────────────

class TestVWAPSignal:
    def test_no_signal_in_first_bars(self):
        sig = VWAPSignal(std_threshold=1.5)
        for _ in range(3):
            result = sig.on_bar("AAPL", make_bar())
        assert result is None

    def test_far_above_vwap_triggers_fade_short(self):
        sig = VWAPSignal(std_threshold=0.5)   # very tight for test
        base_bar = make_bar(open_=100, high=100.5, low=99.5, close=100, volume=100_000)
        for _ in range(10):
            sig.on_bar("AAPL", base_bar)
        # Now a bar far above VWAP
        high_bar = make_bar(open_=105, high=106, low=104.5, close=105.5, volume=100_000)
        result = sig.on_bar("AAPL", high_bar)
        assert result == "fade_short"

    def test_reset_clears_state(self):
        sig = VWAPSignal()
        for _ in range(5):
            sig.on_bar("AAPL", make_bar())
        sig.reset()
        assert sig.get_vwap("AAPL") is None


# ── IntradayComposite ──────────────────────────────────────────────────────

class TestIntradayComposite:
    def test_rank_candidates_returns_top_n(self):
        composite = IntradayComposite(top_n=3)
        snaps = [
            make_snapshot("A", gap_pct=0.03, volume_ratio=3.0),
            make_snapshot("B", gap_pct=0.02, volume_ratio=3.0),
            make_snapshot("C", gap_pct=0.04, volume_ratio=3.0),
            make_snapshot("D", gap_pct=0.025, volume_ratio=3.0),
            make_snapshot("E", gap_pct=0.035, volume_ratio=3.0),
        ]
        candidates = composite.rank_candidates(snaps)
        assert len(candidates) <= 3

    def test_highest_score_ranked_first(self):
        composite = IntradayComposite(top_n=5)
        snaps = [
            make_snapshot("LOW", gap_pct=0.02, volume_ratio=2.5),
            make_snapshot("HIGH", gap_pct=0.06, volume_ratio=5.0),
        ]
        candidates = composite.rank_candidates(snaps)
        tickers = [c.ticker for c in candidates]
        assert tickers.index("HIGH") < tickers.index("LOW")

    def test_set_day_type(self):
        composite = IntradayComposite()
        composite.set_day_type("range")
        assert composite._day_type == "range"


# ── RiskLimits.size_from_risk ──────────────────────────────────────────────

class TestSizeFromRisk:
    def test_basic_sizing(self):
        limits = RiskLimits(max_single_trade_risk=0.005)
        # $100k × 0.5% = $500 risk budget; stop = $1.00 from entry → 500 shares
        qty = limits.size_from_risk(100_000, entry=50.0, stop=49.0, risk_pct=0.005)
        assert qty == 500

    def test_invalid_stop_returns_zero(self):
        limits = RiskLimits()
        assert limits.size_from_risk(100_000, entry=50.0, stop=50.0) == 0

    def test_entry_zero_returns_zero(self):
        limits = RiskLimits()
        assert limits.size_from_risk(100_000, entry=0.0, stop=49.0) == 0

    def test_minimum_quantity_is_one(self):
        limits = RiskLimits()
        # Tiny risk budget should still return at least 1 share
        qty = limits.size_from_risk(100, entry=50.0, stop=0.01, risk_pct=0.005)
        assert qty >= 1


# ── DrawdownMonitor (intraday extensions) ──────────────────────────────────

class TestIntradayDrawdown:
    def test_daily_pnl_halt_triggers(self):
        dd = DrawdownMonitor(daily_pnl_halt_pct=-0.02)
        dd.reset_daily(100_000)
        assert not dd.daily_pnl_halted
        dd.update_intraday(97_500)   # -2.5%
        assert dd.daily_pnl_halted

    def test_daily_pnl_no_halt_above_threshold(self):
        dd = DrawdownMonitor(daily_pnl_halt_pct=-0.02)
        dd.reset_daily(100_000)
        dd.update_intraday(99_500)   # -0.5%
        assert not dd.daily_pnl_halted

    def test_reset_daily_clears_halt(self):
        dd = DrawdownMonitor(daily_pnl_halt_pct=-0.02)
        dd.reset_daily(100_000)
        dd.update_intraday(97_000)   # triggers halt
        assert dd.daily_pnl_halted
        dd.reset_daily(97_000)
        assert not dd.daily_pnl_halted


# ── PositionManager ────────────────────────────────────────────────────────

class TestPositionManager:
    def _make_manager(self):
        broker = MagicMock()
        broker.submit_bracket_order.return_value = ("order-1", "stop-1", "target-1")
        broker.market_sell_all_intraday.return_value = 0
        return PositionManager(broker=broker, max_concurrent=3)

    def _make_order(self, ticker="AAPL", side=Side.BUY):
        return BracketOrder(
            ticker=ticker,
            side=side,
            qty=100,
            entry_price=150.0,
            stop_price=149.0,
            target_price=152.0,
        )

    def test_enter_adds_position(self):
        pm = self._make_manager()
        pm.try_enter(self._make_order("AAPL"))
        assert pm.has_position("AAPL")
        assert pm.open_count == 1

    def test_duplicate_entry_rejected(self):
        pm = self._make_manager()
        pm.try_enter(self._make_order("AAPL"))
        result = pm.try_enter(self._make_order("AAPL"))
        assert not result
        assert pm.open_count == 1

    def test_max_concurrent_respected(self):
        pm = self._make_manager()
        for ticker in ["A", "B", "C"]:
            pm.try_enter(self._make_order(ticker))
        assert pm.open_count == 3
        result = pm.try_enter(self._make_order("D"))
        assert not result
        assert pm.open_count == 3

    def test_hard_close_empties_positions(self):
        pm = self._make_manager()
        pm.try_enter(self._make_order("AAPL"))
        pm.hard_close_all()
        assert pm.open_count == 0

    def test_hard_close_logs_to_closed_today(self):
        pm = self._make_manager()
        pm.try_enter(self._make_order("AAPL"))
        pm.hard_close_all()
        assert len(pm.closed_today) == 1
        assert pm.closed_today[0]["close_type"] == "hard_close"

    def test_low_rr_order_rejected(self):
        pm = self._make_manager()
        # 0.5:1 R:R — below 1.5 minimum
        bad_order = BracketOrder(
            ticker="AAPL",
            side=Side.BUY,
            qty=100,
            entry_price=150.0,
            stop_price=149.0,
            target_price=150.50,    # only 0.5:1
        )
        result = pm.try_enter(bad_order)
        assert not result
        assert pm.open_count == 0

    def test_reset_day_clears_all_state(self):
        pm = self._make_manager()
        pm.try_enter(self._make_order("AAPL"))
        pm.reset_day()
        assert pm.open_count == 0
        assert len(pm.closed_today) == 0
