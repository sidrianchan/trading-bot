"""Audit when SPY 200d MA trend filter SHOULD have fired vs when it DID fire."""
from __future__ import annotations

import yfinance as yf
import pandas as pd

print("Fetching SPY 2017-2024...")
spy = yf.download("SPY", start="2017-01-01", end="2025-01-01", progress=False, auto_adjust=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy = spy["Close"]

ma200 = spy.rolling(200).mean()
below = spy < ma200

# Month-end aligned (last trading day of each month) — when rebalance happens
month_ends = spy.resample("ME").last().index
month_ends = month_ends[month_ends >= "2018-01-01"]

# Engine actually fired on these dates (from backtest log):
fired = {
    pd.Timestamp("2018-10-31"), pd.Timestamp("2019-05-31"),
    pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-31"),
    pd.Timestamp("2020-04-30"), pd.Timestamp("2023-10-31"),
}

print(f"\n{'Date':<12} {'SPY':>8} {'200dMA':>8} {'Below?':>7} {'EngineFired?':>14}")
print("-" * 55)

# Find the actual trading day on or before each month-end and compare
for me in month_ends:
    if me not in spy.index:
        idx = spy.index[spy.index <= me]
        if len(idx) == 0:
            continue
        actual_date = idx[-1]
    else:
        actual_date = me
    px = spy.loc[actual_date]
    ma = ma200.loc[actual_date] if actual_date in ma200.index else float("nan")
    is_below = bool(px < ma) if pd.notna(ma) else False
    fired_here = actual_date in fired
    flag_should = "BELOW" if is_below else ""
    flag_did    = "FIRED" if fired_here else ""
    discrepancy = " <-- MISSED" if (is_below and not fired_here) else (" <-- SPURIOUS" if (not is_below and fired_here) else "")
    print(f"{actual_date.date()!s:<12} {px:>8.2f} {ma:>8.2f} {flag_should:>7} {flag_did:>14}{discrepancy}")

# Continuous below-MA periods — what's the longest stretch we should have been in cash?
print("\n\nContinuous SPY-below-200dMA periods (gap closes after >5 consecutive days back above):")
print(f"{'Start':<12} {'End':<12} {'Days':>5} {'SPY @start':>10} {'SPY @end':>10} {'Drawdown':>10}")
print("-" * 65)
in_below = False
start_date = None
peak_before = None
for d, b in below.items():
    if pd.isna(b):
        continue
    if b and not in_below:
        in_below = True
        start_date = d
        # SPY peak in last 60 days before crossing
        window = spy.loc[d - pd.Timedelta(days=60):d]
        peak_before = window.max() if len(window) else spy.loc[d]
    elif not b and in_below:
        in_below = False
        days = (d - start_date).days
        if days >= 10:  # only show meaningful stretches
            spy_start = spy.loc[start_date]
            spy_end = spy.loc[d]
            min_during = spy.loc[start_date:d].min()
            dd = (min_during - peak_before) / peak_before
            print(f"{start_date.date()!s:<12} {d.date()!s:<12} {days:>5d} {spy_start:>10.2f} {spy_end:>10.2f} {dd:>9.1%}")
