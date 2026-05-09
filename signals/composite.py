from __future__ import annotations

import pandas as pd
from loguru import logger

from signals.base import BaseSignal
from signals.momentum import MomentumSignal
from signals.quality import QualitySignal
from signals.volatility import LowVolatilitySignal


class CompositeSignal(BaseSignal):
    """Weighted combination of momentum, quality, and low-volatility signals.

    Architecture is designed to accept an XGBoost ranker as a fourth signal
    (or as a replacement) once sufficient training data is available.
    Each sub-signal is converted to a percentile rank before weighting,
    so scores are on a common [0, 1] scale before combination.
    """

    def __init__(
        self,
        momentum_weight: float = 0.60,
        quality_weight: float = 0.30,
        low_vol_weight: float = 0.10,
        lookback_days: int = 252,
        skip_days: int = 21,
        vol_lookback_days: int = 63,
        top_n: int = 25,
    ):
        self.top_n = top_n
        self.weights = {
            "momentum": momentum_weight,
            "quality": quality_weight,
            "low_vol": low_vol_weight,
        }
        self._signals: dict[str, BaseSignal] = {
            "momentum": MomentumSignal(lookback_days, skip_days),
            "quality": QualitySignal(),
            "low_vol": LowVolatilitySignal(vol_lookback_days),
        }

    def register_signal(self, name: str, signal: BaseSignal, weight: float) -> None:
        """Add a signal at an absolute final weight; existing signals are scaled
        proportionally to sum to (1 - weight).

        Example: register_signal('ml', xgb_signal, weight=0.60) gives ML 60%
        and shrinks all prior signals to share the remaining 40%.
        """
        if not 0 < weight < 1:
            raise ValueError(f"weight must be in (0, 1), got {weight}")
        existing_total = sum(v for k in self.weights for v in [self.weights[k]] if k != name)
        scale = (1.0 - weight) / existing_total if existing_total > 0 else 0.0
        self.weights = {k: v * scale for k, v in self.weights.items() if k != name}
        self.weights[name] = weight
        self._signals[name] = signal
        logger.info(
            f"Registered signal '{name}' at {weight:.0%}. "
            f"Final weights: { {k: f'{v:.0%}' for k, v in sorted(self.weights.items())} }"
        )

    def compute(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
        composite = pd.Series(dtype=float)

        for name, signal in self._signals.items():
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            ranks = signal.rank(prices, fundamentals)
            if ranks.empty:
                logger.debug(f"Signal '{name}' returned empty — skipping")
                continue
            if composite.empty:
                composite = w * ranks
            else:
                composite = composite.add(w * ranks, fill_value=0.0)

        if composite.empty:
            return composite

        # Only return top_n stocks; set others to NaN so portfolio ignores them
        threshold = composite.nlargest(self.top_n).min()
        composite[composite < threshold] = float("nan")
        return composite.dropna()
