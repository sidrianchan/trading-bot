"""Integration tests for backtester/ta_engine.py."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.ta_engine import (
    TABacktester,
    TABacktestConfig,
    _aggregate_by_trade,
    _ClosedLeg,
    _OpenTrade,
)
from signals.setup import SetupEngineConfig, TradeSetup


def _setup(ticker: str = "X", **overrides) -> TradeSetup:
    base = dict(
        ticker=ticker,
        direction="long",
        hold_type="intraday",
        entry_price=100.0,
        stop_price=99.0,
        target1_price=101.0,
        target2_price=103.0,
        atr=0.50,
        score=80.0,
        components={"sr_level": 25, "candlestick": 20, "trend_alignment": 20,
                    "rsi": 15, "macd": 0, "volume": 0},
        sr_level=99.0,
        pattern_detail="hammer",
        setup_date=date(2024, 6, 1),
    )
    base.update(overrides)
    return TradeSetup(**base)


def _bar(o, h, l, c, v=1_000):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


# ── Aggregation ────────────────────────────────────────────────────────────


class TestAggregateByTrade:
    def test_single_trade_two_legs(self):
        legs = [
            _ClosedLeg("X", "long", "swing", entry_dt=pd.Timestamp("2024-06-01 09:30").to_pydatetime(),
                       exit_dt=pd.Timestamp("2024-06-01 10:30").to_pydatetime(),
                       entry_price=100, exit_price=101, qty=10, pnl=10.0, close_type="t1"),
            _ClosedLeg("X", "long", "swing", entry_dt=pd.Timestamp("2024-06-01 09:30").to_pydatetime(),
                       exit_dt=pd.Timestamp("2024-06-01 11:30").to_pydatetime(),
                       entry_price=100, exit_price=103, qty=10, pnl=30.0, close_type="t2"),
        ]
        out = _aggregate_by_trade(legs)
        assert len(out) == 1
        assert out[0]["pnl"] == 40.0
        assert out[0]["exit_dt"] == legs[1].exit_dt

    def test_two_separate_trades(self):
        legs = [
            _ClosedLeg("A", "long", "intraday",
                       entry_dt=pd.Timestamp("2024-06-01 10:00").to_pydatetime(),
                       exit_dt=pd.Timestamp("2024-06-01 14:00").to_pydatetime(),
                       entry_price=100, exit_price=99, qty=10, pnl=-10.0, close_type="stop"),
            _ClosedLeg("B", "long", "intraday",
                       entry_dt=pd.Timestamp("2024-06-01 11:00").to_pydatetime(),
                       exit_dt=pd.Timestamp("2024-06-01 15:00").to_pydatetime(),
                       entry_price=50, exit_price=52, qty=20, pnl=40.0, close_type="t2"),
        ]
        out = _aggregate_by_trade(legs)
        assert len(out) == 2


# ── Sizing & risk caps ─────────────────────────────────────────────────────


class TestSizingAndRisk:
    def test_size_respects_one_percent_risk(self):
        # max_position_value_pct=1.0 disables the value cap so we isolate the
        # risk-based sizing rule.
        cfg = TABacktestConfig(initial_capital=100_000.0, risk_per_trade_pct=0.01,
                                max_position_value_pct=1.0)
        bt = TABacktester(cfg, SetupEngineConfig())
        # entry 100, stop 99 → risk per share 1.00 → expected qty = 1000
        s = _setup(entry_price=100.0, stop_price=99.0)
        assert bt._size(s) == 1000

    def test_size_capped_when_position_value_binding(self):
        # 10% of $100k = $10k cap; $100 entry → 100 shares max even though
        # risk-based size (1000 shares) would risk only $1k (= 1% of port)
        cfg = TABacktestConfig(initial_capital=100_000.0, risk_per_trade_pct=0.01,
                                max_position_value_pct=0.10)
        bt = TABacktester(cfg, SetupEngineConfig())
        s = _setup(entry_price=100.0, stop_price=99.0)
        assert bt._size(s) == 100

    def test_size_capped_by_position_value(self):
        cfg = TABacktestConfig(initial_capital=100_000.0, risk_per_trade_pct=0.01,
                                max_position_value_pct=0.10)
        bt = TABacktester(cfg, SetupEngineConfig())
        # tiny stop → "1% risk" sizing produces enormous qty; cap at $10k value
        s = _setup(entry_price=100.0, stop_price=99.99)  # risk = $0.01 → 100k shares
        # 10% of 100k = $10k cap → 100 shares max
        assert bt._size(s) == 100

    def test_size_zero_when_stop_too_tight(self):
        cfg = TABacktestConfig()
        bt = TABacktester(cfg, SetupEngineConfig())
        s = _setup(entry_price=100.0, stop_price=99.999)  # risk < $0.01
        assert bt._size(s) == 0

    def test_can_open_blocked_by_max_concurrent(self):
        cfg = TABacktestConfig(max_concurrent=2)
        bt = TABacktester(cfg, SetupEngineConfig())
        bt._open["A"] = _OpenTrade(setup=_setup("A"), entry_dt=pd.Timestamp.now().to_pydatetime(),
                                    qty=10, qty_remaining=10, trailing_stop=99.0)
        bt._open["B"] = _OpenTrade(setup=_setup("B"), entry_dt=pd.Timestamp.now().to_pydatetime(),
                                    qty=10, qty_remaining=10, trailing_stop=99.0)
        assert not bt._can_open("C")

    def test_can_open_blocked_by_sector_cap(self):
        cfg = TABacktestConfig(max_concurrent=10, max_per_sector=2)
        sectors = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Health"}
        bt = TABacktester(cfg, SetupEngineConfig(), sectors=sectors)
        bt._open["A"] = _OpenTrade(setup=_setup("A"), entry_dt=pd.Timestamp.now().to_pydatetime(),
                                    qty=10, qty_remaining=10, trailing_stop=99.0,
                                    sector="Tech")
        bt._open["B"] = _OpenTrade(setup=_setup("B"), entry_dt=pd.Timestamp.now().to_pydatetime(),
                                    qty=10, qty_remaining=10, trailing_stop=99.0,
                                    sector="Tech")
        assert not bt._can_open("C")    # third Tech blocked
        assert bt._can_open("D")        # different sector ok


# ── Trade lifecycle (T1 → T2 → trail) ──────────────────────────────────────


class TestTradeLifecycle:
    def _bt_with_open_long(self, capital=100_000.0):
        cfg = TABacktestConfig(initial_capital=capital, target1_pct=0.40, target2_pct=0.40,
                                trail_atr_mult=1.0, spread_bps=0)  # zero spread → clean math
        bt = TABacktester(cfg, SetupEngineConfig())
        s = _setup(entry_price=100.0, stop_price=99.0,
                   target1_price=101.0, target2_price=103.0, atr=0.50)
        ot = _OpenTrade(setup=s, entry_dt=pd.Timestamp("2024-06-01 10:00").to_pydatetime(),
                         qty=100, qty_remaining=100, trailing_stop=99.0)
        bt._open["X"] = ot
        return bt, ot

    def test_t1_partial_fill_moves_stop_to_breakeven(self):
        bt, ot = self._bt_with_open_long()
        bar = _bar(100.5, 101.5, 100.0, 101.2)
        ts = pd.Timestamp("2024-06-01 10:30")
        bt._manage_open_trade(ot, bar, ts, lambda px: 0.0)
        assert ot.t1_filled
        assert ot.qty_remaining == 60
        assert ot.trailing_stop == 100.0   # breakeven

    def test_t2_partial_then_trailing(self):
        bt, ot = self._bt_with_open_long()
        # Trigger T1
        bt._manage_open_trade(ot, _bar(100.5, 101.5, 100.0, 101.2),
                               pd.Timestamp("2024-06-01 10:30"), lambda p: 0.0)
        # Trigger T2
        bt._manage_open_trade(ot, _bar(101.0, 103.5, 100.5, 103.2),
                               pd.Timestamp("2024-06-01 11:30"), lambda p: 0.0)
        assert ot.t2_filled
        assert ot.qty_remaining == 20
        # Trailing kicks in: close=103.2, atr=0.5 → trail=102.7
        assert ot.trailing_stop == pytest.approx(102.7, abs=1e-9)

    def test_stop_hit_closes_trade(self):
        bt, ot = self._bt_with_open_long()
        bar = _bar(100.0, 100.2, 98.5, 98.8)   # low pierces stop=99.0
        bt._manage_open_trade(ot, bar, pd.Timestamp("2024-06-01 10:30"), lambda p: 0.0)
        assert ot.qty_remaining == 0
        assert any(c.close_type == "stop" for c in bt._closed)

    def test_short_trade_t1_and_stop(self):
        cfg = TABacktestConfig(initial_capital=100_000.0, spread_bps=0)
        bt = TABacktester(cfg, SetupEngineConfig())
        s = _setup(direction="short", entry_price=100.0, stop_price=101.0,
                   target1_price=99.0, target2_price=97.0, atr=0.5)
        ot = _OpenTrade(setup=s, entry_dt=pd.Timestamp("2024-06-01 10:00").to_pydatetime(),
                         qty=100, qty_remaining=100, trailing_stop=101.0)
        bt._open["X"] = ot
        bt._manage_open_trade(ot, _bar(100, 100.2, 98.5, 98.8),
                               pd.Timestamp("2024-06-01 10:30"), lambda p: 0.0)
        assert ot.t1_filled
        assert ot.qty_remaining == 60
        assert ot.trailing_stop == 100.0   # short breakeven = entry


# ── Gate report (smoke) ────────────────────────────────────────────────────


class TestGateReport:
    def test_report_no_trades_does_not_crash(self, capsys):
        from backtester.ta_engine import _report
        cfg = TABacktestConfig()
        equity = pd.Series([100_000.0], index=pd.to_datetime(["2024-01-02"]))
        _report(cfg, equity, [], label="empty")
        out = capsys.readouterr().out
        assert "GATE FAILED" in out  # 0 trades → win-rate 0% → fails
