from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class PortfolioConstructor:
    """Inverse-volatility position sizing with hard caps and sector limits.

    Stocks are weighted proportional to 1/volatility so that each position
    contributes roughly equally to portfolio risk. Weights are then normalized
    and clipped to the configured min/max bounds.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        max_position_size: float = 0.10,
        min_position_size: float = 0.01,
        max_sector_exposure: float = 0.30,
        vol_lookback_days: int = 63,
    ):
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_sector_exposure = max_sector_exposure
        self.vol_lookback_days = vol_lookback_days

    def construct(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Convert signal scores to portfolio weights.

        Returns a Series of weights (sum <= 1.0, remainder stays as cash).
        Returns an empty Series to signal "go to cash."
        """
        if scores.empty:
            return pd.Series(dtype=float)

        tickers = scores.index.tolist()
        available = [t for t in tickers if t in prices.columns]
        if not available:
            return pd.Series(dtype=float)

        # Compute realized volatility for each stock
        recent_prices = prices[available].iloc[-self.vol_lookback_days :]
        returns = recent_prices.pct_change().dropna()
        ann_vol = (returns.std() * np.sqrt(252)).replace(0, np.nan)

        # Inverse-volatility weights
        inv_vol = (1.0 / ann_vol).dropna()
        raw_weights = inv_vol / inv_vol.sum()

        # Apply position size caps
        weights = raw_weights.clip(lower=self.min_position_size, upper=self.max_position_size)
        weights = weights / weights.sum()  # renormalize after clipping

        # Apply sector concentration limit
        if fundamentals is not None and "sector" in fundamentals.columns:
            weights = self._apply_sector_limit(weights, fundamentals)

        # Scale to target portfolio volatility
        port_vol = self._estimate_portfolio_vol(weights, returns)
        if port_vol > 0:
            scale = self.target_volatility / port_vol
            weights = weights * min(scale, 1.0)

        # Re-apply caps after scaling (vol rescale can breach max)
        weights = self._clip_and_normalize(weights)

        logger.debug(
            f"Constructed portfolio: {len(weights)} positions, "
            f"total weight={weights.sum():.2%}, "
            f"max={weights.max():.2%}, min={weights.min():.2%}"
        )
        return weights

    def _apply_sector_limit(
        self, weights: pd.Series, fundamentals: pd.DataFrame
    ) -> pd.Series:
        sectors = fundamentals["sector"].reindex(weights.index)
        for sector, group in sectors.groupby(sectors):
            tickers_in_sector = group.index
            sector_weight = weights[tickers_in_sector].sum()
            if sector_weight > self.max_sector_exposure:
                scale = self.max_sector_exposure / sector_weight
                weights[tickers_in_sector] *= scale

        total = weights.sum()
        return weights / total if total > 0 else weights

    def _clip_and_normalize(self, weights: pd.Series) -> pd.Series:
        """Iteratively clip to [min, max] and renormalize until stable."""
        for _ in range(20):
            clipped = weights.clip(lower=self.min_position_size, upper=self.max_position_size)
            total = clipped.sum()
            normalized = clipped / total if total > 0 else clipped
            if (normalized - weights).abs().max() < 1e-8:
                return normalized
            weights = normalized
        return weights

    @staticmethod
    def _estimate_portfolio_vol(weights: pd.Series, returns: pd.DataFrame) -> float:
        common = weights.index.intersection(returns.columns)
        if common.empty:
            return 0.0
        w = weights[common].values
        cov = returns[common].cov().values * 252
        variance = float(w @ cov @ w)
        return float(np.sqrt(max(variance, 0)))
