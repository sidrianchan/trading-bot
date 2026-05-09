"""Tests for risk management components."""
import pytest
import pandas as pd

from risk.drawdown import DrawdownMonitor
from risk.kill_switch import KillSwitch
from risk.limits import RiskLimits


class TestDrawdownMonitor:
    def test_no_drawdown_at_start(self):
        monitor = DrawdownMonitor(limit=0.15)
        monitor.update(1000.0)
        assert monitor.current_drawdown(1000.0) == pytest.approx(0.0)
        assert not monitor.is_halted

    def test_circuit_breaker_triggers_at_limit(self):
        monitor = DrawdownMonitor(limit=0.15, reset_threshold=0.07)
        monitor.update(1000.0)   # establish peak
        monitor.update(840.0)    # 16% drawdown — should trigger
        assert monitor.is_halted

    def test_circuit_breaker_does_not_trigger_below_limit(self):
        monitor = DrawdownMonitor(limit=0.15)
        monitor.update(1000.0)
        monitor.update(900.0)    # 10% drawdown — should NOT trigger
        assert not monitor.is_halted

    def test_circuit_breaker_resets_on_recovery(self):
        monitor = DrawdownMonitor(limit=0.15, reset_threshold=0.07)
        monitor.update(1000.0)
        monitor.update(840.0)    # trigger
        assert monitor.is_halted
        monitor.update(940.0)    # 6% dd — below reset threshold
        assert not monitor.is_halted

    def test_peak_tracks_highest_value(self):
        monitor = DrawdownMonitor()
        monitor.update(1000.0)
        monitor.update(1200.0)
        monitor.update(1100.0)
        assert monitor.peak == pytest.approx(1200.0)

    def test_drawdown_calculation_correct(self):
        monitor = DrawdownMonitor()
        monitor.update(1000.0)
        dd = monitor.current_drawdown(850.0)
        assert dd == pytest.approx(0.15, rel=1e-3)


class TestKillSwitch:
    def test_not_triggered_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ks = KillSwitch()
        assert not ks.is_triggered()

    def test_trigger_creates_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ks = KillSwitch()
        ks.trigger("test reason")
        assert ks.is_triggered()
        assert ks.reason() == "test reason"

    def test_reset_removes_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ks = KillSwitch()
        ks.trigger("halt")
        ks.reset()
        assert not ks.is_triggered()

    def test_reset_when_not_triggered_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ks = KillSwitch()
        ks.reset()  # should not raise
        assert not ks.is_triggered()


class TestRiskLimits:
    def test_weights_clipped_to_max(self):
        limits = RiskLimits(max_position_size=0.10)
        weights = pd.Series({"A": 0.20, "B": 0.05, "C": 0.08})
        result = limits.apply(weights)
        assert result["A"] <= 0.10 + 1e-9

    def test_weights_capped_excess_becomes_cash(self):
        # RiskLimits clips to max; excess becomes cash (sum < 1 is intentional).
        # Renormalization happens upstream in PortfolioConstructor, not here.
        limits = RiskLimits(max_position_size=0.10)
        weights = pd.Series({"A": 0.50, "B": 0.50})
        result = limits.apply(weights)
        assert result["A"] == pytest.approx(0.10)
        assert result["B"] == pytest.approx(0.10)
        assert result.sum() == pytest.approx(0.20)

    def test_empty_input_returns_empty(self):
        limits = RiskLimits()
        result = limits.apply(pd.Series(dtype=float))
        assert result.empty

    def test_trade_within_risk_limit(self):
        limits = RiskLimits(max_single_trade_risk=0.02)
        # $500 trade with 15% stop on $10,000 portfolio = 0.75% risk — OK
        assert limits.check_trade_risk("AAPL", 500.0, 10000.0, stop_loss_pct=0.15)

    def test_trade_exceeds_risk_limit(self):
        limits = RiskLimits(max_single_trade_risk=0.02)
        # $2000 trade with 15% stop on $10,000 portfolio = 3% risk — exceeds limit
        assert not limits.check_trade_risk("AAPL", 2000.0, 10000.0, stop_loss_pct=0.15)
