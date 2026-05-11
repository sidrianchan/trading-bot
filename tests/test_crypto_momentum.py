import pandas as pd
import pytest

from signals.crypto_momentum import (
    CryptoMomentumConfig,
    CryptoMomentumState,
    compute_crypto_signal,
    normalize_crypto_symbol,
    total_return_skip,
)


def test_normalize_crypto_symbol_variants():
    assert normalize_crypto_symbol("BTC-USD") == "BTC/USD"
    assert normalize_crypto_symbol("BTCUSD") == "BTC/USD"
    assert normalize_crypto_symbol("ETH/USD") == "ETH/USD"


def test_total_return_skip_uses_skipped_window_end():
    series = pd.Series(range(1, 21), dtype=float)
    result = total_return_skip(series, lookback=5, skip=2)
    assert result == pytest.approx(series.iloc[-3] / series.iloc[-8] - 1.0)


def test_risk_off_when_btc_absolute_momentum_negative():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    prices = pd.DataFrame(
        {
            "BTC/USD": list(reversed(range(100, 220))),
            "ETH/USD": list(reversed(range(200, 320))),
        },
        index=dates,
    )
    cfg = CryptoMomentumConfig(abs_lookback=84, abs_skip=14, rel_lookback=7, rel_skip=14)
    signal, _ = compute_crypto_signal(prices, CryptoMomentumState(peak=30_000), 30_000, cfg)
    assert signal.target is None
    assert signal.regime == "risk_off"


def test_risk_on_selects_stronger_relative_asset():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    btc = pd.Series(range(100, 220), index=dates, dtype=float)
    eth = btc.copy()
    eth.iloc[-22:-14] = [220, 225, 230, 235, 240, 245, 250, 255]
    prices = pd.DataFrame({"BTC/USD": btc, "ETH/USD": eth})
    cfg = CryptoMomentumConfig(abs_lookback=84, abs_skip=14, rel_lookback=7, rel_skip=14)
    signal, _ = compute_crypto_signal(prices, CryptoMomentumState(peak=30_000), 30_000, cfg)
    assert signal.regime == "risk_on"
    assert signal.target == "ETH/USD"


def test_circuit_breaker_forces_usdc_when_holding_risk_asset():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    prices = pd.DataFrame(
        {"BTC/USD": range(100, 220), "ETH/USD": range(100, 220)},
        index=dates,
        dtype=float,
    )
    cfg = CryptoMomentumConfig(cb_threshold=-0.40)
    state = CryptoMomentumState(peak=100_000, last_target="BTC/USD")
    signal, new_state = compute_crypto_signal(prices, state, 55_000, cfg)
    assert signal.regime == "circuit_breaker"
    assert signal.target is None
    assert new_state.last_target is None
