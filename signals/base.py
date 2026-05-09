from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSignal(ABC):
    """Abstract base for all signal generators.

    Concrete implementations return a score Series indexed by ticker.
    Higher score = more attractive. The interface is deliberately simple
    so an XGBoost ranker can be swapped in as a drop-in replacement.
    """

    @abstractmethod
    def compute(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
        """Compute raw signal scores.

        Args:
            prices: Adjusted close prices, shape (dates, tickers).
            fundamentals: Fundamental data, shape (tickers, fields).

        Returns:
            Series indexed by ticker, higher = more attractive. May contain NaN
            for tickers with insufficient data.
        """

    def rank(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
        """Return percentile ranks [0, 1] — higher = more attractive."""
        scores = self.compute(prices, fundamentals)
        return scores.rank(pct=True, na_option="bottom")
