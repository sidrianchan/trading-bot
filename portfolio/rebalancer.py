from __future__ import annotations

import calendar
from datetime import date

import pandas as pd
from loguru import logger


class Rebalancer:
    """Determines when to rebalance and generates the delta of weights to trade.

    Only rebalances on the last trading day of each month. Between rebalances,
    positions are held unless a stop-loss is triggered by the risk layer.
    """

    def __init__(self, drift_threshold: float = 0.20):
        # Rebalance early if any position drifts >20% relative to its target
        self.drift_threshold = drift_threshold
        self._last_rebalance: date | None = None
        self._target_weights: pd.Series = pd.Series(dtype=float)

    def should_rebalance(self, today: date, prices: pd.DataFrame) -> bool:
        if self._last_rebalance is None:
            return True
        if _is_last_trading_day_of_month(today, prices):
            return True
        if not self._target_weights.empty:
            # Emergency rebalance on significant drift
            return False  # drift check happens in live loop via risk layer
        return False

    def record_rebalance(self, today: date, target_weights: pd.Series) -> None:
        self._last_rebalance = today
        self._target_weights = target_weights.copy()

    def generate_orders(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        min_trade_size: float = 0.005,
    ) -> pd.Series:
        """Return signed weight deltas: positive = buy, negative = sell.

        Trades smaller than min_trade_size are suppressed to avoid tiny orders.
        """
        all_tickers = target_weights.index.union(current_weights.index)
        target = target_weights.reindex(all_tickers, fill_value=0.0)
        current = current_weights.reindex(all_tickers, fill_value=0.0)

        delta = target - current
        delta = delta[delta.abs() >= min_trade_size]

        if not delta.empty:
            logger.info(
                f"Rebalance: {(delta > 0).sum()} buys, {(delta < 0).sum()} sells"
            )
        return delta


def _is_last_trading_day_of_month(today: date, prices: pd.DataFrame) -> bool:
    """True if today is the last date in prices for this calendar month."""
    this_month = prices.index[
        (prices.index.month == today.month) & (prices.index.year == today.year)
    ]
    if this_month.empty:
        return False
    return pd.Timestamp(today) >= this_month[-1]
