"""Per-year strategy vs SPY breakdown from /tmp/backtest_equity.csv."""
from __future__ import annotations

import pandas as pd

df = pd.read_csv("/tmp/backtest_equity.csv", index_col=0, parse_dates=True)
df["year"] = df.index.year

print(f"\n{'Year':<6} {'Strat ret':>10} {'SPY ret':>10} {'Alpha':>8} {'Beat SPY':>10}")
print("-" * 50)

beat_count = 0
total_count = 0
for year, group in df.groupby("year"):
    s_start, s_end = group["strategy"].iloc[0], group["strategy"].iloc[-1]
    b_start, b_end = group["spy"].iloc[0], group["spy"].iloc[-1]
    s_ret = (s_end / s_start) - 1
    b_ret = (b_end / b_start) - 1
    alpha = s_ret - b_ret
    beat = "YES" if s_ret > b_ret else "no"
    if s_ret > b_ret:
        beat_count += 1
    total_count += 1
    print(f"{year:<6} {s_ret:>9.1%} {b_ret:>9.1%} {alpha:>+7.1%} {beat:>10}")

print("-" * 50)
print(f"{'TOTAL':<6}  Strategy beat SPY in {beat_count}/{total_count} calendar years")
