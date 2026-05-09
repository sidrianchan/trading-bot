from __future__ import annotations

import pandas as pd
from loguru import logger


class RiskLimits:
    """Enforces hard position-size and per-trade risk limits.

    Applied as a final gate before orders are generated. Operates on
    target weights; any weight that would violate a limit is clipped.
    """

    def __init__(
        self,
        max_position_size: float = 0.10,
        max_single_trade_risk: float = 0.02,
    ):
        self.max_position_size = max_position_size
        self.max_single_trade_risk = max_single_trade_risk

    def apply(self, weights: pd.Series) -> pd.Series:
        """Clip positions to the hard max. Does NOT renormalize.

        Renormalization belongs in PortfolioConstructor. RiskLimits is a hard
        guard — excess weight becomes cash, not redistributed to other positions.
        """
        if weights.empty:
            return weights

        trimmed = (weights > self.max_position_size).sum()
        if trimmed > 0:
            logger.debug(f"Risk limits capping {trimmed} position(s) to {self.max_position_size:.0%}")

        return weights.clip(upper=self.max_position_size)

    def size_from_risk(
        self,
        portfolio_value: float,
        entry: float,
        stop: float,
        risk_pct: float | None = None,
    ) -> int:
        """Return share quantity sized so that (entry − stop) × qty = portfolio × risk_pct.

        Returns 0 if the stop is at or beyond the entry (invalid setup).
        """
        pct = risk_pct if risk_pct is not None else self.max_single_trade_risk
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0 or entry <= 0:
            return 0
        dollar_risk = portfolio_value * pct
        return max(1, int(dollar_risk / risk_per_share))

    def check_trade_risk(
        self,
        ticker: str,
        trade_notional: float,
        portfolio_value: float,
        stop_loss_pct: float = 0.15,
    ) -> bool:
        """Return True if the trade's risk (notional * stop_loss) is within the limit."""
        risk_amount = trade_notional * stop_loss_pct
        risk_pct = risk_amount / portfolio_value if portfolio_value > 0 else 1.0
        if risk_pct > self.max_single_trade_risk:
            logger.warning(
                f"{ticker}: trade risk {risk_pct:.2%} exceeds limit {self.max_single_trade_risk:.2%}"
            )
            return False
        return True
