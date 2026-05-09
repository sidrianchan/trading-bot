from __future__ import annotations

import pandas as pd

from signals.base import BaseSignal


class QualitySignal(BaseSignal):
    """Composite quality factor: ROE, low debt, revenue growth.

    Higher score = higher quality. Stocks must pass basic data availability
    requirements; tickers with all-NaN fundamentals are dropped.
    """

    def __init__(
        self,
        roe_weight: float = 0.40,
        debt_equity_weight: float = 0.30,
        revenue_growth_weight: float = 0.30,
    ):
        self.roe_weight = roe_weight
        self.debt_equity_weight = debt_equity_weight
        self.revenue_growth_weight = revenue_growth_weight

    def compute(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=fundamentals.index)
        total_weight = 0.0

        if "roe" in fundamentals.columns:
            rank = fundamentals["roe"].rank(pct=True, na_option="bottom")
            score += self.roe_weight * rank
            total_weight += self.roe_weight

        if "debt_equity" in fundamentals.columns:
            # Lower debt/equity is better, so invert the rank
            rank = 1.0 - fundamentals["debt_equity"].rank(pct=True, na_option="top")
            score += self.debt_equity_weight * rank
            total_weight += self.debt_equity_weight

        if "revenue_growth" in fundamentals.columns:
            rank = fundamentals["revenue_growth"].rank(pct=True, na_option="bottom")
            score += self.revenue_growth_weight * rank
            total_weight += self.revenue_growth_weight

        if total_weight == 0:
            return pd.Series(dtype=float)

        return (score / total_weight).dropna()
