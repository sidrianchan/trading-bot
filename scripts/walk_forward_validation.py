"""Rolling walk-forward validation for the ETF and crypto momentum strategies.

Rewritten 2026-09-05 after an audit found the previous version reported 91.1% CAGR
for the crypto strategy when the honest figure was negative. The design was sound;
what it left out was decisive. Three changes:

1. **Every free parameter is re-selected inside each training window.** The old
   version re-selected two (``abs_lookback``, ``cb_threshold``) and held the rest
   fixed at values chosen by looking at the whole history. ``rel_lookback`` alone
   was worth 91.1% at its fixed value of 7 against 34-69% elsewhere — fitted on
   data the walk-forward then pretended not to have seen.

2. **Rebalance timing is tranched, not selected.** Which weekday (crypto) or which
   day of the month (ETF) has no economic content, and it carried the whole result:
   91.1% on Monday, 1.2% on Saturday. Selecting the best is fitting; averaging all
   of them is the answer. Each window runs every tranche at 1/N capital.

3. **Costs are charged and a benchmark is printed next to every figure.** Alpaca's
   crypto taker fee is 25 bps and the strategy trades ~52 legs a year — 12.2% a
   year. The old grid never paid it, and never printed what BTC did over the same
   window.

The stitched out-of-sample curve is the result. Nothing else on this page is.

Usage: python scripts/walk_forward_validation.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Running this as `python scripts/walk_forward_validation.py` puts scripts/ on the
# path, not the repo root, so the package imports below fail. The documented usage
# should just work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtester.costs import ALPACA_CRYPTO_TAKER_BPS, ETF_SPREAD_SLIPPAGE_BPS
from backtester.crypto_momentum import fetch_crypto_prices, run_crypto_tranched
from backtester.dual_momentum import fetch_etf_prices, run_dual_momentum_tranched
from signals.crypto_momentum import CryptoMomentumConfig
from signals.dual_momentum import V4Config

ETF_CAPITAL = 10_000.0

# Every dimension the strategy is free in gets re-chosen each window. Anything left
# out of this product is, by definition, fitted on the full history.
ETF_GRID = [
    V4Config(abs_lookback=a, rel_lookback=r, cb_threshold=c, reentry_confirmation_months=m)
    for a, r, c, m in itertools.product((126, 189, 252), (42, 63, 126), (0.25, 0.40), (0, 2))
]
CRYPTO_GRID = [
    CryptoMomentumConfig(abs_lookback=a, rel_lookback=r, cb_threshold=c)
    for a, r, c in itertools.product((56, 84, 112), (7, 14, 28), (-0.30, -0.40, -0.50))
]


def metrics(equity: pd.Series, periods_per_year: int) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol > 0 else float("nan")
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "vol": float(vol),
        "max_dd": float((equity / equity.cummax() - 1.0).min()),
    }


def rolling_windows(index: pd.DatetimeIndex, start_year: int, month: int, train_years: int, test_years: int):
    windows, year = [], start_year
    while True:
        train_start = pd.Timestamp(year=year, month=month, day=1)
        train_end = train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        if test_start > index[-1]:
            return windows
        windows.append((train_start, train_end, test_start, min(test_end, index[-1])))
        year += test_years


def walk_forward(
    prices: pd.DataFrame,
    grid: list,
    runner,
    label: str,
    periods_per_year: int,
    start_year: int,
    month: int,
    train_years: int,
    test_years: int,
    cost_bps: float,
    benchmarks: dict[str, str],
    capital: float,
    runner_kwargs: dict | None = None,
) -> None:
    runner_kwargs = runner_kwargs or {}
    print(f"\n{'=' * 92}")
    print(f"  {label} — ROLLING WALK-FORWARD  (train {train_years}y / test {test_years}y, "
          f"{cost_bps:.0f} bps/leg, timing tranched)")
    print("=" * 92)

    windows = rolling_windows(prices.index, start_year, month, train_years, test_years)
    oos_returns, total_legs, leg_years = [], 0.0, 0.0

    for train_start, train_end, test_start, test_end in windows:
        scored = []
        for cfg in grid:
            r = runner(prices, cfg, str(train_start.date()), str(train_end.date()),
                       cost_bps=cost_bps, **runner_kwargs)
            m = metrics(r.equity, periods_per_year)
            if not np.isnan(m["sharpe"]):
                scored.append((m["sharpe"], cfg))
        if not scored:
            print(f"  {train_start.date()}–{train_end.date()}: no viable config, window skipped")
            continue
        best = max(scored, key=lambda x: x[0])[1]

        # Re-run from train_start so the strategy carries its warm-up state into the
        # test slice, then keep only the test slice's returns.
        full = runner(prices, best, str(train_start.date()), str(test_end.date()),
                      cost_bps=cost_bps, **runner_kwargs)
        window_returns = full.equity.pct_change().dropna().loc[str(test_start.date()):str(test_end.date())]
        oos_returns.append(window_returns)
        # Count legs over the span they were actually counted over (train + test),
        # not over the stitched test-only window — otherwise turnover reads high by
        # the train/test ratio.
        total_legs += full.turnover.get("legs", 0.0)
        leg_years += full.turnover.get("years", 0.0)

        m = metrics(full.equity.loc[str(test_start.date()):], periods_per_year)
        params = {k: v for k, v in vars(best).items() if not isinstance(v, tuple) and not isinstance(v, str)}
        print(f"  Test {test_start.date()}–{test_end.date()}  "
              f"{params}  →  CAGR {m['cagr']:+7.1%}  Sharpe {m['sharpe']:5.2f}  MaxDD {m['max_dd']:7.1%}")

    if not oos_returns:
        print("  No out-of-sample windows produced.")
        return

    stitched = capital * (1.0 + pd.concat(oos_returns)).cumprod()
    m = metrics(stitched, periods_per_year)
    lo, hi = stitched.index[0], stitched.index[-1]

    print(f"\n  {'STITCHED OUT-OF-SAMPLE':<28} {lo.date()}–{hi.date()}")
    print(f"  {'':<28} CAGR {m['cagr']:+7.1%}   Sharpe {m['sharpe']:5.2f}   "
          f"Vol {m['vol']:5.1%}   MaxDD {m['max_dd']:7.1%}")
    print(f"  {'':<28} {total_legs / leg_years:.0f} legs/yr per tranche "
          f"({total_legs / leg_years * (1 if leg_years else 0):.0f} round-trips ≈ "
          f"{100 * (1 - (1 - cost_bps / 1e4) ** (total_legs / leg_years)):.1f}%/yr in costs)")

    print(f"\n  {'BENCHMARKS, same window':<28}")
    for name, column in benchmarks.items():
        if column not in prices.columns:
            continue
        bench = prices[column].loc[lo:hi].reindex(stitched.index).ffill()
        bench = bench / bench.iloc[0] * capital
        b = metrics(bench, periods_per_year)
        verdict = "strategy WINS" if b["sharpe"] < m["sharpe"] else "strategy LOSES"
        print(f"  {name:<28} CAGR {b['cagr']:+7.1%}   Sharpe {b['sharpe']:5.2f}   "
              f"Vol {b['vol']:5.1%}   MaxDD {b['max_dd']:7.1%}   → {verdict} on Sharpe")

        # The comparison that actually decides it. Any strategy can beat a benchmark
        # on return by taking more risk, or beat it on risk by holding cash. The
        # honest question is whether it beats *the same benchmark held at the same
        # volatility, with no trading at all* — a position anyone can hold for free.
        if b["vol"] > 0:
            w = m["vol"] / b["vol"]
            scaled = capital * (1.0 + w * bench.pct_change().fillna(0.0)).cumprod()
            s = metrics(scaled, periods_per_year)
            edge = "adds nothing" if m["sharpe"] <= s["sharpe"] + 0.02 else "adds Sharpe"
            dd_edge = (m["max_dd"] - s["max_dd"]) * 100  # in percentage points
            print(f"  {f'  └─ {w:.0%} {column}, never traded':<28} CAGR {s['cagr']:+7.1%}   "
                  f"Sharpe {s['sharpe']:5.2f}   Vol {s['vol']:5.1%}   MaxDD {s['max_dd']:7.1%}   "
                  f"→ strategy {edge}; drawdown {dd_edge:+.1f}pp")


def main() -> None:
    etf_prices = fetch_etf_prices()
    walk_forward(
        etf_prices, ETF_GRID, run_dual_momentum_tranched, "ETF STRATEGY", 252,
        start_year=2010, month=3, train_years=6, test_years=2,
        cost_bps=ETF_SPREAD_SLIPPAGE_BPS,
        benchmarks={"SPY buy & hold": "SPY", "TQQQ buy & hold": "TQQQ"},
        capital=ETF_CAPITAL,
        runner_kwargs={"initial_capital": ETF_CAPITAL},
    )

    crypto_prices = fetch_crypto_prices()
    walk_forward(
        crypto_prices, CRYPTO_GRID, run_crypto_tranched, "CRYPTO STRATEGY", 365,
        start_year=2018, month=1, train_years=3, test_years=1,
        cost_bps=ALPACA_CRYPTO_TAKER_BPS,
        benchmarks={"BTC buy & hold": "BTC/USD", "ETH buy & hold": "ETH/USD"},
        capital=CryptoMomentumConfig().capital,
    )

    print(f"\n{'=' * 92}")
    print("  Read the STITCHED line, not the per-window lines. A per-window figure is one draw;")
    print("  the stitched curve is the strategy. And read it against the benchmark beneath it —")
    print("  a CAGR with nothing next to it is not a result.")
    print("=" * 92)


if __name__ == "__main__":
    main()
