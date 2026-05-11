from __future__ import annotations

import os
import time
from typing import Iterable

import pandas as pd
from loguru import logger

from execution.orders import Order, Side
from signals.crypto_momentum import normalize_crypto_symbol


class AlpacaBroker:
    """Thin wrapper around alpaca-py for order submission and account queries.

    Paper trading is the default and safe mode. Live trading requires
    ALPACA_PAPER=false AND explicit confirmation before switching.
    """

    def __init__(self):
        from alpaca.trading.client import TradingClient  # type: ignore
        from alpaca.trading.enums import AssetClass, OrderSide, OrderType, TimeInForce  # type: ignore
        from alpaca.trading.requests import GetAssetsRequest, LimitOrderRequest, MarketOrderRequest  # type: ignore

        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        self._GetAssetsRequest = GetAssetsRequest
        self._OrderSide = OrderSide
        self._OrderType = OrderType
        self._TimeInForce = TimeInForce
        self._AssetClass = AssetClass

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
        self._api_key = api_key
        self._secret_key = secret_key
        self._crypto_data = None
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
            return pd.DataFrame(
                columns=["ticker", "raw_symbol", "normalized_symbol", "qty", "market_value", "weight"]
            )
        rows = []
        for p in positions:
            raw_symbol = str(p.symbol)
            rows.append(
                {
                    "ticker": raw_symbol,
                    "raw_symbol": raw_symbol,
                    "normalized_symbol": normalize_crypto_symbol(raw_symbol),
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry": float(p.avg_entry_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                }
            )
        return pd.DataFrame(rows).set_index("ticker")

    def get_positions_for_symbols(self, symbols: Iterable[str]) -> pd.DataFrame:
        normalized = {normalize_crypto_symbol(s) for s in symbols}
        positions = self.get_positions()
        if positions.empty:
            return positions
        scoped = positions[positions["normalized_symbol"].isin(normalized)].copy()
        if scoped.empty:
            return scoped
        return scoped.set_index("normalized_symbol", drop=False)

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
                f"${order.notional:,.2f} -> id={result.id}"
            )
            return str(result.id)
        except Exception as exc:
            logger.error(f"Order failed for {order}: {exc}")
            return None

    def submit_crypto_limit_order(
        self,
        symbol: str,
        side: Side,
        limit_price: float,
        notional: float | None = None,
        qty: float | None = None,
    ) -> str | None:
        """Submit a GTC crypto limit order. Provide either notional or qty."""
        if (notional is None) == (qty is None):
            raise ValueError("Provide exactly one of notional or qty")
        order_side = self._OrderSide.BUY if side == Side.BUY else self._OrderSide.SELL
        try:
            req = self._LimitOrderRequest(
                symbol=normalize_crypto_symbol(symbol),
                notional=round(notional, 2) if notional is not None else None,
                qty=qty,
                side=order_side,
                type=self._OrderType.LIMIT,
                time_in_force=self._TimeInForce.GTC,
                limit_price=round(limit_price, 2),
            )
            result = self._client.submit_order(req)
            logger.info(
                f"Crypto limit submitted: {side.value.upper()} {symbol} "
                f"notional={notional} qty={qty} limit={limit_price:.2f} id={result.id}"
            )
            return str(result.id)
        except Exception as exc:
            logger.error(f"Crypto limit order failed for {symbol}: {exc}")
            return None

    def wait_for_order_fill(self, order_id: str, timeout_seconds: int, poll_seconds: int = 5):
        deadline = time.monotonic() + timeout_seconds
        last_order = None
        while time.monotonic() < deadline:
            last_order = self._client.get_order_by_id(order_id)
            if str(getattr(last_order, "status", "")).lower() == "filled":
                return last_order
            time.sleep(poll_seconds)
        return last_order

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)
        logger.warning(f"Order cancelled: {order_id}")

    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()
        logger.warning("All open orders cancelled")

    def liquidate_all(self) -> None:
        """Emergency: close all positions at market."""
        logger.critical("LIQUIDATING ALL POSITIONS")
        self._client.close_all_positions(cancel_orders=True)

    def liquidate_symbols(self, symbols: Iterable[str]) -> int:
        """Close only positions whose symbols are in the supplied bot universe."""
        scoped = self.get_positions_for_symbols(symbols)
        if scoped.empty:
            logger.info("Scoped liquidation: no matching positions")
            return 0
        count = 0
        for _, pos in scoped.iterrows():
            raw_symbol = str(pos.get("raw_symbol") or pos.name)
            logger.warning(f"Scoped liquidation: closing {raw_symbol}")
            self._client.close_position(raw_symbol)
            count += 1
        return count

    def get_crypto_asset_metadata(self, symbol: str):
        assets = self._client.get_all_assets(
            self._GetAssetsRequest(asset_class=self._AssetClass.CRYPTO)
        )
        normalized = normalize_crypto_symbol(symbol)
        for asset in assets:
            if normalize_crypto_symbol(str(asset.symbol)) == normalized:
                return asset
        raise ValueError(f"Crypto asset not found: {symbol}")

    def get_crypto_mid_price(self, symbol: str) -> float:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient  # type: ignore
        from alpaca.data.requests import CryptoLatestOrderbookRequest  # type: ignore

        if self._crypto_data is None:
            self._crypto_data = CryptoHistoricalDataClient(self._api_key, self._secret_key)
        normalized = normalize_crypto_symbol(symbol)
        request = CryptoLatestOrderbookRequest(symbol_or_symbols=normalized)
        books = self._crypto_data.get_crypto_latest_orderbook(request)
        book = books[normalized] if isinstance(books, dict) else books
        bid = self._level_price(book.bids[0])
        ask = self._level_price(book.asks[0])
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError(f"Invalid order book for {symbol}: bid={bid}, ask={ask}")
        return (bid + ask) / 2.0

    @staticmethod
    def _level_price(level) -> float:
        if hasattr(level, "price"):
            return float(level.price)
        if isinstance(level, dict):
            return float(level.get("p") or level.get("price"))
        return float(level[0])

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
        from alpaca.trading.requests import StopLossRequest, TakeProfitRequest  # type: ignore

        side = self._OrderSide.BUY if order.side.value == "buy" else self._OrderSide.SELL
        try:
            req = self._LimitOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=self._TimeInForce.DAY,
                type=self._OrderType.LIMIT,
                limit_price=round(order.entry_price, 2),
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=round(order.target_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(order.stop_price, 2)),
            )
            result = self._client.submit_order(req)
            entry_id = str(result.id)
            legs = getattr(result, "legs", []) or []
            stop_id = str(legs[0].id) if len(legs) > 0 else ""
            target_id = str(legs[1].id) if len(legs) > 1 else ""
            logger.info(
                f"Bracket submitted: {order.side.value.upper()} {order.ticker} "
                f"x{order.qty} entry={order.entry_price:.2f} "
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
