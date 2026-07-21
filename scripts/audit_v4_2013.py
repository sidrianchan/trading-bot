"""Detailed audit of V4 (6m + tight CB) through the 2012-2013 MDD episode.

Reuses the run_variant() implementation from dual_momentum_variants.py to
ensure identical logic, then dumps month-by-month trade log + daily equity
curve for 2010-01-01 → 2014-12-31 (covering the full peak-to-trough path).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dual_momentum_variants import (  # type: ignore
    Variant, fetch_prices, run_variant, ALL_ETFS_3X, INITIAL_CAPITAL,
)

import pandas as pd

V4 = Variant("V4 6m+tight", 126, ALL_ETFS_3X, 0.25, 2)


def main() -> None:
    prices = fetch_prices()
    eq, trades = run_variant(prices, V4)

    # Restrict to relevant window
    window_start = pd.Timestamp("2010-01-01")
    window_end = pd.Timestamp("2014-12-31")
    eq_w = eq.loc[window_start:window_end]
    trades_w = [t for t in trades if window_start <= t["date"] <= window_end]

    # Equity curve summary — find peak and trough in window, also print daily DD
    peak_idx = eq_w.idxmax()
    peak_val = eq_w.max()
    cm = eq_w.cummax()
    dd = (eq_w - cm) / cm
    trough_idx = dd.idxmin()
    trough_val = eq_w.loc[trough_idx]
    overall_trough_dd = dd.min()

    print(f"\n{'='*78}")
    print(f"  V4 (6m + tight CB) — 2010-2014 audit")
    print(f"{'='*78}")
    print(f"  Window peak     : {peak_idx.date()}  ${peak_val:>12,.0f}")
    print(f"  Window trough   : {trough_idx.date()}  ${trough_val:>12,.0f}")
    print(f"  In-window MDD   : {overall_trough_dd:>6.1%}")
    print(f"  Trades in window: {len(trades_w)}")
    print()

    print("Month-by-month rebalance log (2012-01-01 to 2014-06-30):")
    print(f"{'Date':<12} {'Held':<8} {'Regime':<12} {'SPY 6m':>8} {'CB?':>5} {'Portfolio':>14}")
    print("-" * 64)
    for t in trades_w:
        if pd.Timestamp("2012-01-01") <= t["date"] <= pd.Timestamp("2014-06-30"):
            spy = f"{t['spy_ret']:>+7.1%}" if pd.notna(t["spy_ret"]) else "    n/a"
            cb_flag = "ON" if t.get("in_cb") else ""
            print(f"{t['date'].date()!s:<12} {t['held']:<8} {t['regime']:<12} "
                  f"{spy:>8} {cb_flag:>5} ${t['portfolio_value']:>12,.0f}")

    # Find the dates the CB actually fired (intraday liquidations) — these aren't
    # in the trade log directly, but the equity curve will show characteristic flat
    # transitions. Also show daily equity at key dates.
    print(f"\n\nDaily equity at key dates:")
    print(f"{'Date':<12} {'Equity':>14} {'DD':>8}")
    print("-" * 36)
    for d in [
        pd.Timestamp("2010-12-31"), pd.Timestamp("2011-04-29"),
        pd.Timestamp("2011-07-29"), pd.Timestamp("2011-09-30"),
        pd.Timestamp("2011-12-30"), pd.Timestamp("2012-03-30"),
        pd.Timestamp("2012-09-28"), pd.Timestamp("2013-01-31"),
        pd.Timestamp("2013-04-18"),  # The MDD trough date from earlier output
        pd.Timestamp("2013-12-31"), pd.Timestamp("2014-12-31"),
    ]:
        if d in eq.index:
            v = eq.loc[d]
            d_dd = dd.loc[d] if d in dd.index else float("nan")
            print(f"{d.date()!s:<12} ${v:>12,.0f} {d_dd:>7.1%}")

    # Save full equity curve for the window
    eq_w.to_csv("/tmp/v4_equity_2010_2014.csv", header=["equity"])
    print(f"\nFull window equity curve → /tmp/v4_equity_2010_2014.csv")


if __name__ == "__main__":
    main()
