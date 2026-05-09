from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from execution.orders import BracketOrder, OpenTrade, Side

if TYPE_CHECKING:
    from execution.broker import AlpacaBroker


class PositionManager:
    """Tracks all open intraday bracket trades.

    Responsibilities:
    - Accepts new BracketOrders and submits them via the broker
    - Tracks partial exits (50% off at 1:1 reward:risk)
    - Enforces the hard 15:45 ET close — calls broker.market_sell_all_intraday()
    - Exposes open trade state to the agent loop for reporting

    Position sizing is done upstream (by RiskLimits.size_from_risk).
    This class manages the lifecycle of each trade after entry.
    """

    def __init__(self, broker: "AlpacaBroker", max_concurrent: int = 8):
        self._broker = broker
        self._max_concurrent = max_concurrent
        self._trades: dict[str, OpenTrade] = {}   # ticker → OpenTrade
        self._closed_today: list[dict] = []       # trade results for daily report

    def reset_day(self) -> None:
        """Call at 09:25 ET. Clears state from previous day."""
        self._trades.clear()
        self._closed_today.clear()

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def try_enter(self, order: BracketOrder) -> bool:
        """Submit a bracket order if within concurrent position limits.

        Returns True if the order was submitted, False if skipped.
        """
        if order.ticker in self._trades:
            logger.debug(f"{order.ticker}: already have an open position, skipping")
            return False

        if len(self._trades) >= self._max_concurrent:
            logger.debug(
                f"Max concurrent positions ({self._max_concurrent}) reached; "
                f"skipping {order.ticker}"
            )
            return False

        if order.reward_risk_ratio < 1.5:
            logger.debug(
                f"{order.ticker}: R:R {order.reward_risk_ratio:.2f} < 1.5 minimum; skipping"
            )
            return False

        entry_id, stop_id, target_id = self._broker.submit_bracket_order(order)
        if not entry_id:
            return False

        trade = OpenTrade(
            ticker=order.ticker,
            side=order.side,
            qty=order.qty,
            entry_price=order.entry_price,
            stop_price=order.stop_price,
            target_price=order.target_price,
            half_qty=order.qty // 2,
            alpaca_entry_id=entry_id,
            alpaca_stop_id=stop_id,
            alpaca_target_id=target_id,
        )
        self._trades[order.ticker] = trade
        logger.info(
            f"Entered: {order.side.value.upper()} {order.ticker} ×{order.qty} "
            f"@ {order.entry_price:.2f} | stop={order.stop_price:.2f} "
            f"target={order.target_price:.2f} | R:R={order.reward_risk_ratio:.2f}"
        )
        return True

    # ------------------------------------------------------------------
    # Per-bar management
    # ------------------------------------------------------------------

    def on_bar(self, ticker: str, bar: dict) -> None:
        """Check partial-exit conditions on each incoming bar.

        Alpaca bracket orders manage the stop and target automatically.
        This method only handles the 1:1 partial exit (50% close at midpoint).
        """
        trade = self._trades.get(ticker)
        if trade is None or trade.partial_closed:
            return

        price = bar["close"]
        direction = 1 if trade.side == Side.BUY else -1
        move = direction * (price - trade.entry_price)
        one_to_one = trade.risk_per_share  # equal to R, so 1:1 reward = 1R

        if move >= one_to_one and trade.half_qty > 0:
            self._partial_close(trade, price)

    def _partial_close(self, trade: OpenTrade, price: float) -> None:
        """Market-sell half the position at 1:1 reward."""
        from alpaca.trading.requests import MarketOrderRequest  # type: ignore
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore

        exit_side = OrderSide.SELL if trade.side == Side.BUY else OrderSide.BUY
        try:
            req = MarketOrderRequest(
                symbol=trade.ticker,
                qty=trade.half_qty,
                side=exit_side,
                time_in_force=TimeInForce.DAY,
            )
            self._broker._client.submit_order(req)
            trade.partial_closed = True
            pnl = (price - trade.entry_price) * trade.half_qty * (1 if trade.side == Side.BUY else -1)
            logger.info(
                f"{trade.ticker}: partial exit ×{trade.half_qty} @ {price:.2f} "
                f"(1:1 hit) P&L=${pnl:+.2f}"
            )
        except Exception as exc:
            logger.error(f"Partial close failed for {trade.ticker}: {exc}")

    # ------------------------------------------------------------------
    # Hard close (15:45 ET)
    # ------------------------------------------------------------------

    def hard_close_all(self) -> int:
        """Cancel all open orders and market-sell all positions. No exceptions."""
        n = self._broker.market_sell_all_intraday()
        for ticker in list(self._trades):
            trade = self._trades.pop(ticker)
            self._closed_today.append({
                "ticker": ticker,
                "side": trade.side.value,
                "qty": trade.qty,
                "entry": trade.entry_price,
                "close_type": "hard_close",
                "opened_at": str(trade.opened_at),
            })
        logger.warning(f"Hard close complete: {n} position(s) liquidated at 15:45 ET")
        return n

    def record_closed(self, ticker: str, exit_price: float, close_type: str) -> None:
        """Called externally when a bracket leg fills (stop or target hit)."""
        trade = self._trades.pop(ticker, None)
        if trade is None:
            return
        direction = 1 if trade.side == Side.BUY else -1
        pnl = direction * (exit_price - trade.entry_price) * trade.qty
        result = {
            "ticker": ticker,
            "side": trade.side.value,
            "qty": trade.qty,
            "entry": trade.entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "close_type": close_type,
            "closed_at": datetime.now(),
            "hold_minutes": int((datetime.now() - trade.opened_at).total_seconds() / 60),
        }
        self._closed_today.append(result)
        logger.info(
            f"Closed: {ticker} {close_type} @ {exit_price:.2f} "
            f"P&L=${pnl:+.2f} (held {result['hold_minutes']}m)"
        )

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def open_count(self) -> int:
        return len(self._trades)

    @property
    def open_trades(self) -> list[OpenTrade]:
        return list(self._trades.values())

    @property
    def closed_today(self) -> list[dict]:
        return list(self._closed_today)

    def has_position(self, ticker: str) -> bool:
        return ticker in self._trades
