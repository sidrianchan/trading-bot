from __future__ import annotations

from loguru import logger


class DrawdownMonitor:
    """Tracks peak portfolio value and triggers a circuit breaker.

    When the drawdown from peak exceeds `limit`, trading halts.
    Resumes only when the drawdown recovers below `reset_threshold`.
    """

    def __init__(self, limit: float = 0.15, reset_threshold: float = 0.07,
                 daily_pnl_halt_pct: float = -0.02):
        self.limit = limit
        self.reset_threshold = reset_threshold
        self.daily_pnl_halt_pct = daily_pnl_halt_pct
        self._peak: float = 0.0
        self._halted: bool = False
        self._daily_start: float = 0.0
        self._daily_halted: bool = False

    def update(self, portfolio_value: float) -> None:
        if portfolio_value > self._peak:
            self._peak = portfolio_value

        dd = self.current_drawdown(portfolio_value)

        if not self._halted and dd >= self.limit:
            self._halted = True
            logger.warning(
                f"Circuit breaker ACTIVATED: drawdown {dd:.1%} >= limit {self.limit:.1%}. "
                "Rebalancing suspended."
            )

        if self._halted and dd <= self.reset_threshold:
            self._halted = False
            logger.info(
                f"Circuit breaker RESET: drawdown recovered to {dd:.1%}. "
                "Rebalancing resumed."
            )

    def current_drawdown(self, portfolio_value: float) -> float:
        if self._peak == 0:
            return 0.0
        return max(0.0, (self._peak - portfolio_value) / self._peak)

    def reset_daily(self, start_value: float) -> None:
        """Call at 09:25 ET each morning to arm the intraday P&L halt."""
        self._daily_start = start_value
        self._daily_halted = False

    def update_intraday(self, current_value: float) -> None:
        """Check current P&L against the daily halt threshold."""
        if self._daily_start <= 0 or self._daily_halted:
            return
        pnl_pct = (current_value - self._daily_start) / self._daily_start
        if pnl_pct <= self.daily_pnl_halt_pct:
            self._daily_halted = True
            logger.warning(
                f"Intraday halt: daily P&L {pnl_pct:.1%} ≤ {self.daily_pnl_halt_pct:.1%}. "
                "No new entries until tomorrow."
            )

    def daily_pnl_pct(self, current_value: float) -> float:
        if self._daily_start <= 0:
            return 0.0
        return (current_value - self._daily_start) / self._daily_start

    @property
    def daily_pnl_halted(self) -> bool:
        return self._daily_halted

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def peak(self) -> float:
        return self._peak
