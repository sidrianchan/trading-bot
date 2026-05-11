from types import SimpleNamespace

import pandas as pd
import pytest

from execution.broker import AlpacaBroker
from execution.orders import Side


class FakeClient:
    def __init__(self):
        self.submitted = []
        self.cancelled = []
        self.closed = []
        self.order = SimpleNamespace(status="filled", filled_avg_price="100")

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id="order-1")

    def get_order_by_id(self, order_id):
        return self.order

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(order_id)

    def get_all_positions(self):
        return [
            SimpleNamespace(symbol="BTCUSD", qty="0.2", market_value="20000", avg_entry_price="90000", unrealized_pl="100"),
            SimpleNamespace(symbol="TQQQ", qty="10", market_value="1000", avg_entry_price="90", unrealized_pl="5"),
        ]

    def close_position(self, symbol):
        self.closed.append(symbol)


def make_broker():
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._client = FakeClient()
    broker._OrderSide = SimpleNamespace(BUY="buy", SELL="sell")
    broker._OrderType = SimpleNamespace(LIMIT="limit")
    broker._TimeInForce = SimpleNamespace(GTC="gtc")

    class FakeLimitOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    broker._LimitOrderRequest = FakeLimitOrderRequest
    return broker


def test_get_positions_for_symbols_normalizes_crypto_pairs():
    broker = make_broker()
    scoped = broker.get_positions_for_symbols(["BTC/USD"])
    assert list(scoped.index) == ["BTC/USD"]
    assert scoped.iloc[0]["raw_symbol"] == "BTCUSD"


def test_submit_crypto_limit_order_requires_one_sizing_field():
    broker = make_broker()
    with pytest.raises(ValueError):
        broker.submit_crypto_limit_order("BTC/USD", Side.BUY, 100.0)
    with pytest.raises(ValueError):
        broker.submit_crypto_limit_order("BTC/USD", Side.BUY, 100.0, notional=10, qty=1)


def test_submit_crypto_limit_order_uses_limit_gtc():
    broker = make_broker()
    order_id = broker.submit_crypto_limit_order("BTC/USD", Side.BUY, 100.0, notional=50)
    req = broker._client.submitted[-1]
    assert order_id == "order-1"
    assert req.symbol == "BTC/USD"
    assert req.type == "limit"
    assert req.time_in_force == "gtc"
    assert req.limit_price == 100.0


def test_liquidate_symbols_is_scoped():
    broker = make_broker()
    count = broker.liquidate_symbols(["BTC/USD"])
    assert count == 1
    assert broker._client.closed == ["BTCUSD"]
