from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    ticker: str
    side: Side
    notional: float    # dollar amount (supports fractional shares)
    order_type: str = "market"
    reason: str = ""

    def __repr__(self) -> str:
        return f"Order({self.side.value.upper()} {self.ticker} ${self.notional:,.2f})"


@dataclass
class BracketOrder:
    """Intraday bracket: entry limit + stop + profit target submitted as OCO."""

    ticker: str
    side: Side
    qty: int
    entry_price: float     # limit price for entry
    stop_price: float      # hard stop (below entry for longs, above for shorts)
    target_price: float    # profit target (2:1 R:R minimum)
    reason: str = ""

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def reward_per_share(self) -> float:
        return abs(self.target_price - self.entry_price)

    @property
    def reward_risk_ratio(self) -> float:
        return self.reward_per_share / self.risk_per_share if self.risk_per_share > 0 else 0.0


@dataclass
class OpenTrade:
    """Tracks a live bracket position intraday."""

    ticker: str
    side: Side
    qty: int
    entry_price: float
    stop_price: float
    target_price: float
    half_qty: int                 # quantity to close at 1:1 (partial exit)
    partial_closed: bool = False  # True after 50% taken off at 1:1
    alpaca_entry_id: str = ""
    alpaca_stop_id: str = ""
    alpaca_target_id: str = ""
    opened_at: datetime = field(default_factory=datetime.now)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)

    def unrealized_pnl(self, current_price: float) -> float:
        direction = 1 if self.side == Side.BUY else -1
        remaining = self.qty - (self.half_qty if self.partial_closed else 0)
        return direction * (current_price - self.entry_price) * remaining


def generate_rebalance_orders(
    current_weights: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    min_notional: float = 1.0,
) -> list[Order]:
    """Convert weight deltas into a list of orders.

    Sells are generated first so that cash is available for buys.
    Orders below min_notional are suppressed (Alpaca minimum is $1).
    """
    all_tickers = target_weights.index.union(current_weights.index)
    target = target_weights.reindex(all_tickers, fill_value=0.0)
    current = current_weights.reindex(all_tickers, fill_value=0.0)
    delta = target - current

    orders: list[Order] = []

    # Sells first
    for ticker, dw in delta[delta < 0].items():
        notional = abs(dw) * portfolio_value
        if notional >= min_notional:
            orders.append(Order(ticker=ticker, side=Side.SELL, notional=notional))

    # Then buys
    for ticker, dw in delta[delta > 0].items():
        notional = dw * portfolio_value
        if notional >= min_notional:
            orders.append(Order(ticker=ticker, side=Side.BUY, notional=notional))

    return orders
