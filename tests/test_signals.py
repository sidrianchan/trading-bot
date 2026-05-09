"""Tests for signal generation logic."""
import numpy as np
import pandas as pd
import pytest

from signals.momentum import MomentumSignal
from signals.quality import QualitySignal
from signals.volatility import LowVolatilitySignal
from signals.composite import CompositeSignal


def make_prices(tickers: list[str], n_days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = {}
    for ticker in tickers:
        returns = rng.normal(0.0005, 0.015, n_days)
        prices = 100 * np.cumprod(1 + returns)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


class TestMomentumSignal:
    def test_returns_series_indexed_by_ticker(self):
        prices = make_prices(["AAPL", "MSFT", "GOOG"], n_days=300)
        signal = MomentumSignal(lookback_days=252, skip_days=21)
        scores = signal.compute(prices, pd.DataFrame())
        assert isinstance(scores, pd.Series)
        assert set(scores.index).issubset({"AAPL", "MSFT", "GOOG"})

    def test_insufficient_data_returns_empty(self):
        prices = make_prices(["AAPL"], n_days=100)
        signal = MomentumSignal(lookback_days=252, skip_days=21)
        scores = signal.compute(prices, pd.DataFrame())
        assert scores.empty

    def test_strong_uptrend_scores_higher_than_flat(self):
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        # TREND: strong uptrend over 252 days
        trend = pd.Series(np.linspace(100, 200, 300), index=dates, name="TREND")
        # FLAT: no movement
        flat = pd.Series(np.full(300, 100.0), index=dates, name="FLAT")
        prices = pd.concat([trend, flat], axis=1)

        signal = MomentumSignal(lookback_days=252, skip_days=21)
        scores = signal.compute(prices, pd.DataFrame())
        assert scores["TREND"] > scores["FLAT"]

    def test_rank_produces_values_in_0_1(self):
        prices = make_prices(["A", "B", "C", "D", "E"], n_days=300)
        signal = MomentumSignal()
        ranks = signal.rank(prices, pd.DataFrame())
        assert (ranks >= 0).all() and (ranks <= 1).all()


class TestQualitySignal:
    def make_fundamentals(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "roe": [0.30, 0.05, 0.15],
                "debt_equity": [0.20, 2.00, 0.80],
                "revenue_growth": [0.20, -0.05, 0.10],
            },
            index=["HIGH_QUALITY", "LOW_QUALITY", "MID_QUALITY"],
        )

    def test_high_quality_scores_highest(self):
        signal = QualitySignal()
        scores = signal.compute(pd.DataFrame(), self.make_fundamentals())
        assert scores["HIGH_QUALITY"] > scores["LOW_QUALITY"]

    def test_missing_columns_handled_gracefully(self):
        fund = pd.DataFrame({"roe": [0.2, 0.1]}, index=["A", "B"])
        signal = QualitySignal()
        scores = signal.compute(pd.DataFrame(), fund)
        assert not scores.empty

    def test_all_missing_returns_empty(self):
        signal = QualitySignal()
        scores = signal.compute(pd.DataFrame(), pd.DataFrame())
        assert scores.empty


class TestLowVolatilitySignal:
    def test_low_vol_stock_scores_higher(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        rng = np.random.default_rng(0)
        low_vol = pd.Series(100 + rng.normal(0, 0.5, 100).cumsum(), index=dates, name="LOW")
        high_vol = pd.Series(100 + rng.normal(0, 5.0, 100).cumsum(), index=dates, name="HIGH")
        prices = pd.concat([low_vol, high_vol], axis=1)

        signal = LowVolatilitySignal(lookback_days=63)
        scores = signal.compute(prices, pd.DataFrame())
        assert scores["LOW"] > scores["HIGH"]


class TestCompositeSignal:
    def test_returns_at_most_top_n_stocks(self):
        prices = make_prices([f"T{i}" for i in range(50)], n_days=300)
        fund = pd.DataFrame(
            {"roe": np.random.rand(50), "debt_equity": np.random.rand(50)},
            index=prices.columns,
        )
        signal = CompositeSignal(top_n=10)
        scores = signal.compute(prices, fund)
        assert len(scores) <= 10

    def test_empty_when_insufficient_price_history(self):
        prices = make_prices(["A", "B"], n_days=50)
        signal = CompositeSignal()
        scores = signal.compute(prices, pd.DataFrame())
        assert scores.empty

    def test_register_custom_signal(self):
        from signals.base import BaseSignal

        class ConstantSignal(BaseSignal):
            def compute(self, prices, fundamentals):
                return pd.Series(1.0, index=prices.columns)

        signal = CompositeSignal(top_n=5)
        signal.register_signal("custom", ConstantSignal(), weight=0.5)
        assert "custom" in signal._signals
        assert abs(sum(signal.weights.values()) - 1.0) < 1e-9
