"""Tests for portfolio construction and rebalancing."""
import numpy as np
import pandas as pd
import pytest

from portfolio.construction import PortfolioConstructor
from portfolio.rebalancer import Rebalancer
from execution.orders import generate_rebalance_orders, Side


def make_prices(tickers: list[str], n_days: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    data = {t: 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n_days)) for t in tickers}
    return pd.DataFrame(data, index=dates)


class TestPortfolioConstructor:
    def test_weights_sum_to_at_most_one(self):
        tickers = [f"T{i}" for i in range(10)]
        prices = make_prices(tickers)
        scores = pd.Series(np.random.rand(10), index=tickers)
        constructor = PortfolioConstructor()
        weights = constructor.construct(scores, prices)
        assert weights.sum() <= 1.0 + 1e-9

    def test_no_weight_exceeds_max(self):
        tickers = [f"T{i}" for i in range(10)]
        prices = make_prices(tickers)
        scores = pd.Series(np.ones(10), index=tickers)
        max_pos = 0.10
        constructor = PortfolioConstructor(max_position_size=max_pos)
        weights = constructor.construct(scores, prices)
        # Allow 1 bp floating-point tolerance after iterative normalization
        assert (weights <= max_pos + 1e-4).all()

    def test_no_weight_below_min(self):
        tickers = [f"T{i}" for i in range(5)]
        prices = make_prices(tickers)
        scores = pd.Series(np.ones(5), index=tickers)
        min_pos = 0.02
        constructor = PortfolioConstructor(min_position_size=min_pos)
        weights = constructor.construct(scores, prices)
        assert (weights >= min_pos - 1e-9).all()

    def test_empty_scores_returns_empty(self):
        constructor = PortfolioConstructor()
        weights = constructor.construct(pd.Series(dtype=float), pd.DataFrame())
        assert weights.empty

    def test_lower_vol_stocks_get_higher_weight(self):
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        rng = np.random.default_rng(7)
        low_vol = 100 + rng.normal(0, 0.3, 100).cumsum()
        high_vol = 100 + rng.normal(0, 3.0, 100).cumsum()
        prices = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol}, index=dates)
        scores = pd.Series({"LOW": 1.0, "HIGH": 1.0})
        # Use generous caps so inverse-vol weighting isn't obscured by clipping
        constructor = PortfolioConstructor(max_position_size=0.90, min_position_size=0.0)
        weights = constructor.construct(scores, prices)
        assert weights["LOW"] > weights["HIGH"]


class TestRebalancer:
    def test_generate_orders_sells_before_buys(self):
        current = pd.Series({"A": 0.30, "B": 0.30, "C": 0.40})
        target = pd.Series({"A": 0.10, "B": 0.50, "D": 0.40})
        orders = generate_rebalance_orders(current, target, portfolio_value=10000.0)
        # Sells should come first
        sell_indices = [i for i, o in enumerate(orders) if o.side == Side.SELL]
        buy_indices = [i for i, o in enumerate(orders) if o.side == Side.BUY]
        if sell_indices and buy_indices:
            assert max(sell_indices) < min(buy_indices)

    def test_closed_positions_generate_sell_orders(self):
        current = pd.Series({"A": 0.50, "B": 0.50})
        target = pd.Series({"A": 1.00})  # close B
        orders = generate_rebalance_orders(current, target, portfolio_value=10000.0)
        sell_tickers = {o.ticker for o in orders if o.side == Side.SELL}
        assert "B" in sell_tickers

    def test_small_trades_suppressed(self):
        current = pd.Series({"A": 0.500})
        target = pd.Series({"A": 0.501})  # tiny change
        orders = generate_rebalance_orders(
            current, target, portfolio_value=10000.0, min_notional=5.0
        )
        # $10 trade — should be generated (above $5 min)
        assert len(orders) == 1

        orders_tiny = generate_rebalance_orders(
            current, target, portfolio_value=10.0, min_notional=5.0
        )
        # $0.01 trade — below min, should be suppressed
        assert len(orders_tiny) == 0

    def test_order_notionals_are_positive(self):
        current = pd.Series({"A": 0.30, "B": 0.70})
        target = pd.Series({"A": 0.60, "C": 0.40})
        orders = generate_rebalance_orders(current, target, portfolio_value=5000.0)
        assert all(o.notional > 0 for o in orders)
