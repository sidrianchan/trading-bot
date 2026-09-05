"""Cost model and rebalance-tranching tests.

Added 2026-09-05. These exist because the two bugs they cover — free trading and a
hidden rebalance-timing parameter — between them turned a real -10% CAGR into a
reported +91.1%, and nothing in the suite would have caught either.
"""
import numpy as np
import pandas as pd
import pytest

from backtester.costs import (
    ALPACA_CRYPTO_TAKER_BPS,
    ETF_SPREAD_SLIPPAGE_BPS,
    annual_cost_drag,
    cost_multiplier,
    legs_for_switch,
    turnover_summary,
)
from backtester.crypto_momentum import run_crypto_backtest, run_crypto_tranched
from backtester.dual_momentum import (
    rebalance_indices,
    run_dual_momentum_backtest,
    run_dual_momentum_tranched,
    _is_last_trading_day_of_month,
)
from signals.crypto_momentum import CryptoMomentumConfig
from signals.dual_momentum import V4Config


# ── cost arithmetic ────────────────────────────────────────────────────────────

def test_cost_multiplier_compounds_per_leg():
    assert cost_multiplier(0, 25.0) == 1.0
    assert cost_multiplier(1, 25.0) == pytest.approx(0.9975)
    assert cost_multiplier(2, 25.0) == pytest.approx(0.9975**2)


def test_cost_multiplier_is_identity_when_free():
    assert cost_multiplier(10, 0.0) == 1.0


def test_legs_counts_both_sides_of_a_switch():
    assert legs_for_switch("BTC/USD", "ETH/USD") == 2   # sell one, buy the other
    assert legs_for_switch("BTC/USD", None) == 1        # liquidate to cash
    assert legs_for_switch(None, "BTC/USD") == 1        # deploy from cash
    assert legs_for_switch(None, None) == 0
    assert legs_for_switch("BTC/USD", "BTC/USD") == 0


def test_annual_drag_matches_the_audit_figure():
    """52 weekly legs at Alpaca's taker fee is 12.2% a year — the number that killed
    the crypto strategy. If this ever drifts, the headline finding drifted with it."""
    assert annual_cost_drag(52, ALPACA_CRYPTO_TAKER_BPS) == pytest.approx(0.122, abs=0.001)


def test_turnover_summary_reports_legs_per_year():
    s = turnover_summary(legs=104, years=2.0, cost_bps=ALPACA_CRYPTO_TAKER_BPS)
    assert s["legs_per_year"] == pytest.approx(52.0)
    assert s["annual_cost_drag"] == pytest.approx(0.122, abs=0.001)


# ── rebalance timing ───────────────────────────────────────────────────────────

def test_offset_zero_reproduces_month_end_rule():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    assert rebalance_indices(idx, 0) == {i for i in range(len(idx)) if _is_last_trading_day_of_month(idx, i)}


def test_offset_shifts_every_rebalance_earlier():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    base, shifted = rebalance_indices(idx, 0), rebalance_indices(idx, 3)
    assert shifted == {i - 3 for i in base if i - 3 >= 0}


def test_negative_offset_rejected():
    with pytest.raises(ValueError):
        rebalance_indices(pd.bdate_range("2020-01-01", "2020-03-01"), -1)


# ── engine behaviour on synthetic data ─────────────────────────────────────────

def _crypto_prices(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc = 10_000 * np.exp(np.cumsum(rng.normal(0.0015, 0.03, n)))
    eth = 300 * np.exp(np.cumsum(rng.normal(0.0018, 0.04, n)))
    return pd.DataFrame({"BTC/USD": btc, "ETH/USD": eth}, index=idx)


def _etf_prices(n=900, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    out = {}
    for t, drift, vol in [("SPY", 0.0004, 0.011), ("TQQQ", 0.0009, 0.033),
                          ("UPRO", 0.0008, 0.030), ("SOXL", 0.0010, 0.040), ("TLT", 0.0001, 0.009)]:
        out[t] = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return pd.DataFrame(out, index=idx)


def test_costs_reduce_final_equity_monotonically():
    prices = _crypto_prices()
    s, e = "2020-01-01", "2021-05-01"
    finals = [
        run_crypto_backtest(prices, CryptoMomentumConfig(), s, e, cost_bps=b).equity.iloc[-1]
        for b in (0.0, 10.0, 25.0, 50.0)
    ]
    assert finals == sorted(finals, reverse=True), finals
    assert finals[0] > finals[-1]


def test_default_cost_is_not_free():
    """The regression that matters: a caller who forgets to pass cost_bps must not
    silently get a free backtest."""
    prices = _crypto_prices()
    s, e = "2020-01-01", "2021-05-01"
    default = run_crypto_backtest(prices, CryptoMomentumConfig(), s, e)
    free = run_crypto_backtest(prices, CryptoMomentumConfig(), s, e, cost_bps=0.0)
    assert default.equity.iloc[-1] < free.equity.iloc[-1]
    assert default.turnover["cost_bps_per_leg"] == ALPACA_CRYPTO_TAKER_BPS

    etf = _etf_prices()
    d = run_dual_momentum_backtest(etf, V4Config(), "2018-01-01", "2020-06-01")
    assert d.turnover["cost_bps_per_leg"] == ETF_SPREAD_SLIPPAGE_BPS


def test_turnover_is_reported_and_nonzero():
    r = run_crypto_backtest(_crypto_prices(), CryptoMomentumConfig(), "2020-01-01", "2021-05-01")
    assert r.turnover["legs"] > 0
    assert r.turnover["legs_per_year"] > 0
    assert 0.0 < r.turnover["annual_cost_drag"] < 1.0


def test_rebalance_weekday_changes_the_result():
    """The finding this whole module exists for. If every weekday gave the same
    answer there would be nothing to tranche away."""
    prices = _crypto_prices()
    finals = {
        wd: run_crypto_backtest(prices, CryptoMomentumConfig(), "2020-01-01", "2021-05-01",
                                rebalance_weekday=wd).equity.iloc[-1]
        for wd in range(7)
    }
    assert max(finals.values()) > min(finals.values()) * 1.05


def test_single_tranche_equals_single_run():
    prices = _crypto_prices()
    a = run_crypto_tranched(prices, CryptoMomentumConfig(), "2020-01-01", "2021-05-01", weekdays=(0,))
    b = run_crypto_backtest(prices, CryptoMomentumConfig(), "2020-01-01", "2021-05-01", rebalance_weekday=0)
    pd.testing.assert_series_equal(a.equity, b.equity, check_names=False)


def test_tranched_lands_inside_the_spread_of_its_tranches():
    prices = _crypto_prices()
    s, e = "2020-01-01", "2021-05-01"
    singles = [
        run_crypto_backtest(prices, CryptoMomentumConfig(), s, e, rebalance_weekday=wd).equity.iloc[-1]
        for wd in range(7)
    ]
    blended = run_crypto_tranched(prices, CryptoMomentumConfig(), s, e).equity.iloc[-1]
    assert min(singles) <= blended <= max(singles)


def test_etf_tranched_runs_and_reports_benchmarks():
    prices = _etf_prices()
    r = run_dual_momentum_tranched(prices, V4Config(), "2018-01-01", "2020-06-01", offsets=(0, 1, 2))
    assert "TQQQ buy&hold" in r.summary.index
    assert "SPY buy&hold" in r.summary.index
    assert "Sharpe > TQQQ buy & hold" in r.gates.index
    assert len(r.equity) > 100


def test_empty_tranche_list_rejected():
    with pytest.raises(ValueError):
        run_crypto_tranched(_crypto_prices(), CryptoMomentumConfig(), weekdays=())
    with pytest.raises(ValueError):
        run_dual_momentum_tranched(_etf_prices(), V4Config(), offsets=())


def test_crypto_gates_include_the_benchmark():
    r = run_crypto_backtest(_crypto_prices(), CryptoMomentumConfig(), "2020-01-01", "2021-05-01")
    assert "CAGR > BTC buy & hold" in r.gates.index
