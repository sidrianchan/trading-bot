from __future__ import annotations

import os

import pandas as pd
from loguru import logger

from execution.orders import Order, Side


class AlpacaBroker:
    """Thin wrapper around alpaca-py for order submission and account queries.

    Paper trading is the default and safe mode. Live trading requires
    ALPACA_PAPER=false AND explicit confirmation before switching.
    """

    def __init__(self):
        from alpaca.trading.client import TradingClient  # type: ignore
        from alpaca.trading.requests import MarketOrderRequest  # type: ignore
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore

        self._MarketOrderRequest = MarketOrderRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce

        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

        if not paper:
            logger.warning(
                "LIVE TRADING MODE ACTIVE. Real money is at risk. "
                "Ensure paper trading validation is complete before proceeding."
            )

        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._paper = paper
        logger.info(f"Alpaca broker initialized ({'paper' if paper else 'LIVE'} mode)")

    @property
    def is_paper(self) -> bool:
        return self._paper

    def get_portfolio_value(self) -> float:
        account = self._client.get_account()
        return float(account.portfolio_value)

    def get_cash(self) -> float:
        account = self._client.get_account()
        return float(account.cash)

    def get_positions(self) -> pd.DataFrame:
        positions = self._client.get_all_positions()
        if not positions:
            return pd.DataFrame(columns=["ticker", "qty", "market_value", "weight"])
        rows = [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry": float(p.avg_entry_price),
                "unrealized_pnl": float(p.unrealized_pl),
            }
            for p in positions
        ]
        return pd.DataFrame(rows).set_index("ticker")

    def get_current_weights(self) -> pd.Series:
        positions = self.get_positions()
        if positions.empty:
            return pd.Series(dtype=float)
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return pd.Series(dtype=float)
        return (positions["market_value"] / total_value).rename("weight")

    def submit_order(self, order: Order) -> str | None:
        side = self._OrderSide.BUY if order.side == Side.BUY else self._OrderSide.SELL
        try:
            req = self._MarketOrderRequest(
                symbol=order.ticker,
                notional=round(order.notional, 2),
                side=side,
                time_in_force=self._TimeInForce.CLS,
            )
            result = self._client.submit_order(req)
            logger.info(
                f"Order submitted: {order.side.value.upper()} {order.ticker} "
                f"${order.notional:,.2f} → id={result.id}"
            )
            return str(result.id)
        except Exception as exc:
            logger.error(f"Order failed for {order}: {exc}")
            return None

    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()
        logger.warning("All open orders cancelled")

    def liquidate_all(self) -> None:
        """Emergency: close all positions at market."""
        logger.critical("LIQUIDATING ALL POSITIONS")
        self._client.close_all_positions(cancel_orders=True)

    # ------------------------------------------------------------------
    # Intraday additions
    # ------------------------------------------------------------------

    def get_snapshots(self, tickers: list[str]) -> list:
        """Fetch pre-market snapshots for the morning scan.

        Returns a list of StockSnapshot objects (from signals.gap).
        Delegates to data.streaming.build_snapshots_from_alpaca.
        """
        import os
        from data.streaming import build_snapshots_from_alpaca
        return build_snapshots_from_alpaca(
            tickers,
            os.environ.get("ALPACA_API_KEY", ""),
            os.environ.get("ALPACA_SECRET_KEY", ""),
        )

    def submit_bracket_order(self, order) -> tuple[str, str, str]:
        """Submit an entry limit + stop + target as an Alpaca bracket order.

        Args:
            order: BracketOrder dataclass from execution.orders

        Returns:
            Tuple of (entry_order_id, stop_order_id, target_order_id).
            Empty strings on failure.
        """
        from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest  # type: ignore
        from alpaca.trading.requests import TakeProfitRequest, StopLossRequest   # type: ignore

        side = self._OrderSide.BUY if order.side.value == "buy" else self._OrderSide.SELL
        try:
            req = LimitOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=self._TimeInForce.DAY,
                limit_price=round(order.entry_price, 2),
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=round(order.target_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(order.stop_price, 2)),
            )
            result = self._client.submit_order(req)
            entry_id = str(result.id)
            # Bracket legs are sub-orders; extract their IDs if available
            legs = getattr(result, "legs", []) or []
            stop_id   = str(legs[0].id) if len(legs) > 0 else ""
            target_id = str(legs[1].id) if len(legs) > 1 else ""
            logger.info(
                f"Bracket submitted: {order.side.value.upper()} {order.ticker} "
                f"×{order.qty} entry={order.entry_price:.2f} "
                f"stop={order.stop_price:.2f} target={order.target_price:.2f}"
            )
            return entry_id, stop_id, target_id
        except Exception as exc:
            logger.error(f"Bracket order failed for {order}: {exc}")
            return "", "", ""

    def market_sell_all_intraday(self) -> int:
        """Hard close: market-sell all open positions (15:45 ET rule).

        Returns number of positions closed.
        """
        positions = self._client.get_all_positions()
        if not positions:
            logger.info("Hard close: no open positions")
            return 0

        logger.warning(f"Hard close: liquidating {len(positions)} position(s)")
        self._client.close_all_positions(cancel_orders=True)
        return len(positions)
