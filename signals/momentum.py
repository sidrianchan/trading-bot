from __future__ import annotations

import pandas as pd

from signals.base import BaseSignal


class MomentumSignal(BaseSignal):
    """12-1 month price momentum (Jegadeesh & Titman, 1993).

    Uses the 12-month return but skips the most recent month to avoid
    the short-term reversal effect documented in academic literature.
    """

    def __init__(self, lookback_days: int = 252, skip_days: int = 21):
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def compute(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
        if len(prices) < self.lookback_days + 1:
            return pd.Series(dtype=float)

        # Price at start of lookback window and at skip point
        price_now_proxy = prices.iloc[-(self.skip_days + 1)]   # ~1 month ago
        price_past = prices.iloc[-(self.lookback_days + 1)]    # ~12 months ago

        momentum = (price_now_proxy / price_past) - 1.0
        return momentum.dropna()
