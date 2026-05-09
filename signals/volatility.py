from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import BaseSignal


class LowVolatilitySignal(BaseSignal):
    """Low-volatility anomaly: lower historical volatility = higher score.

    Inverts the volatility rank so that calmer stocks score higher.
    Uses annualized realized volatility over the lookback window.
    """

    def __init__(self, lookback_days: int = 63):
        self.lookback_days = lookback_days

    def compute(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
        if len(prices) < self.lookback_days + 1:
            return pd.Series(dtype=float)

        returns = prices.iloc[-self.lookback_days :].pct_change().dropna()
        ann_vol = returns.std() * np.sqrt(252)

        # Higher score = lower volatility
        inv_vol = (1.0 / ann_vol.replace(0, np.nan)).dropna()
        return inv_vol
