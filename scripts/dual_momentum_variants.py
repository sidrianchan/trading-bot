"""Dual Momentum on Leveraged ETFs — variant comparison.

Runs 5 strategy variants on the same price data, prints a side-by-side
metrics table, and picks the variant that passes all 4 gates with the
best Sharpe.

Variants:
  Original  : 12m filter, -50% CB, immediate re-entry, 3x ETFs (TQQQ/UPRO/SOXL)
  V1a       : 6m  filter, -50% CB, immediate re-entry, 3x ETFs
  V1b       : 3m  filter, -50% CB, immediate re-entry, 3x ETFs
  V2        : 12m filter, -50% CB, immediate re-entry, 2x ETFs (QLD/SSO/USD)
  V3        : 12m filter, -25% CB, 2-month re-entry confirmation, 3x ETFs
  V4        : 6m  filter, -25% CB, 2-month re-entry confirmation, 3x ETFs

Gates:
  CAGR    > 20%
  Max DD  > -60%
  Sharpe  > 0.50
  2022 protective months >= 50%
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

START = "2006-06-21"     # earliest of all ETFs (SSO inception)
END = "2024-12-31"
INITIAL_CAPITAL = 100_000.0
SKIP = 21
REL_LOOKBACK = 63

ALL_ETFS_3X = ["TQQQ", "UPRO", "SOXL"]
ALL_ETFS_2X = ["QLD", "SSO", "USD"]
ALL_TICKERS = sorted(set(ALL_ETFS_3X + ALL_ETFS_2X + ["TLT", "SPY", "QQQ"]))


@dataclass
class Variant:
    name: str
    abs_lookback: int
    risk_on: list[str]
    cb_threshold: float
    reentry_confirmation_months: int


VARIANTS = [
    Variant("Original", 252, ALL_ETFS_3X, 0.50, 0),
    Variant("V1a 6m",   126, ALL_ETFS_3X, 0.50, 0),
    Variant("V1b 3m",    63, ALL_ETFS_3X, 0.50, 0),
    Variant("V2 2x",    252, ALL_ETFS_2X, 0.50, 0),
    Variant("V3 tight", 252, ALL_ETFS_3X, 0.25, 2),
    Variant("V4 6m+tight", 126, ALL_ETFS_3X, 0.25, 2),
    Variant("V5 2x+tight",  252, ALL_ETFS_2X, 0.25, 2),
    Variant("V6 2x+6m+tight", 126, ALL_ETFS_2X, 0.25, 2),
]


def fetch_prices() -> pd.DataFrame:
    print(f"Fetching {ALL_TICKERS} from {START} to {END}...")
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all").ffill()
    # Each ETF starts at its inception date — leave NaN before that, drop rows where
    # all rebalance-relevant tickers are NaN.
    print(f"  → {len(prices)} trading days, columns: {list(prices.columns)}")
    print("\nETF inception dates (first non-NaN):")
    for c in prices.columns:
        first = prices[c].first_valid_index()
        print(f"  {c:<6} {first.date() if first else 'n/a'}")
    return prices


def total_return_skip(series: pd.Series, lookback: int, skip: int) -> float:
    if len(series) < lookback + skip + 1:
        return float("nan")
    end = series.iloc[-(skip + 1)]
    start = series.iloc[-(lookback + skip + 1)]
    if start <= 0 or pd.isna(start) or pd.isna(end):
        return float("nan")
    return end / start - 1.0


def last_trading_days(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    df = pd.DataFrame({"date": index})
    df["ym"] = df["date"].dt.to_period("M")
    return set(df.groupby("ym")["date"].max())


def run_variant(prices: pd.DataFrame, v: Variant) -> tuple[pd.Series, list[dict]]:
    rebalance_dates = last_trading_days(prices.index)
    cash = INITIAL_CAPITAL
    held: str | None = None
    shares: float = 0.0
    peak = INITIAL_CAPITAL
    in_cb = False
    cb_confirm_count = 0  # consecutive risk-on months while CB is up

    daily: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []

    # Effective start: when ALL needed risk-on ETFs and SPY have data
    needed = v.risk_on + ["SPY", "TLT"]
    first_valid = max(prices[t].first_valid_index() for t in needed)
    effective_start = first_valid + pd.Timedelta(days=int((v.abs_lookback + SKIP) * 1.5))

    for d in prices.index:
        row = prices.loc[d]
        equity = shares * row.get(held, 0.0) if held else 0.0
        portfolio_value = cash + equity
        peak = max(peak, portfolio_value)
        dd = (peak - portfolio_value) / peak if peak > 0 else 0.0

        # Intraday CB check (liquidate immediately on threshold breach)
        if dd >= v.cb_threshold and held is not None:
            cash = portfolio_value
            held = None
            shares = 0.0
            in_cb = True
            cb_confirm_count = 0

        if d in rebalance_dates and d >= effective_start:
            history = prices.loc[:d]
            spy_ret = total_return_skip(history["SPY"], v.abs_lookback, SKIP)
            if pd.isna(spy_ret):
                daily[d] = portfolio_value
                continue

            if spy_ret > 0:
                regime = "risk_on"
                candidates = [t for t in v.risk_on if pd.notna(history[t].iloc[-1])]
            else:
                regime = "risk_off"
                tlt_3m = total_return_skip(history["TLT"], REL_LOOKBACK, SKIP)
                candidates = ["TLT"] if (pd.notna(tlt_3m) and tlt_3m > 0) else []

            # Pick top by relative momentum
            target: str | None = None
            if candidates:
                rel = {}
                for t in candidates:
                    r = total_return_skip(history[t], REL_LOOKBACK, SKIP)
                    if pd.notna(r):
                        rel[t] = r
                if rel:
                    target = max(rel, key=rel.get)

            # CB re-entry logic
            if in_cb:
                if regime == "risk_on":
                    cb_confirm_count += 1
                    if cb_confirm_count >= max(1, v.reentry_confirmation_months + 1):
                        in_cb = False
                        peak = portfolio_value  # reset peak; otherwise stuck-in-cash trap
                        cb_confirm_count = 0
                    else:
                        target = None  # remain in cash, awaiting confirmation
                else:
                    cb_confirm_count = 0
                    target = None

            # Switch position
            if held != target:
                if held is not None:
                    cash += shares * row[held]
                    shares = 0.0
                if target is not None and pd.notna(row[target]) and row[target] > 0:
                    shares = cash / row[target]
                    cash = 0.0
                    held = target
                else:
                    held = None

            trades.append({
                "date": d, "held": held or "CASH",
                "regime": regime, "spy_ret": spy_ret,
                "portfolio_value": portfolio_value, "in_cb": in_cb,
            })

        equity_after = shares * row.get(held, 0.0) if held else 0.0
        daily[d] = cash + equity_after

    eq = pd.Series(daily, name=v.name)
    # Trim leading NaN-equivalent days (before effective_start, value just sits at INITIAL)
    eq = eq.loc[eq.index >= first_valid]
    return eq, trades


def metrics(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"),
                "vol": float("nan"), "total_return": float("nan"), "max_dd_2022": float("nan")}
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else float("nan")
    cummax = equity.cummax()
    dd_series = (equity - cummax) / cummax
    max_dd = dd_series.min()
    max_dd_date = dd_series.idxmin()
    # 2022 intra-year max DD
    eq_2022 = equity[equity.index.year == 2022]
    if len(eq_2022) > 1:
        cm = eq_2022.cummax()
        max_dd_2022 = ((eq_2022 - cm) / cm).min()
    else:
        max_dd_2022 = float("nan")
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "vol": vol,
            "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
            "max_dd_2022": max_dd_2022,
            "max_dd_date": max_dd_date}


def protective_pct_2022(trades: list[dict]) -> float:
    rows = [t for t in trades if t["date"].year == 2022]
    if not rows:
        return 0.0
    protective = sum(1 for t in rows if t["held"] in ("CASH", "TLT"))
    return protective / len(rows)


def gate_pass(m: dict, prot_2022: float) -> tuple[bool, list[str]]:
    fails = []
    if m["cagr"] <= 0.20:
        fails.append(f"CAGR {m['cagr']:.1%} ≤ 20%")
    if m["max_dd"] <= -0.60:
        fails.append(f"MDD {m['max_dd']:.1%} worse than -60%")
    if m["sharpe"] <= 0.50:
        fails.append(f"Sharpe {m['sharpe']:.2f} ≤ 0.50")
    if prot_2022 < 0.50:
        fails.append(f"Protective {prot_2022:.0%} < 50%")
    return (len(fails) == 0, fails)


def main() -> None:
    prices = fetch_prices()
    results = []
    for v in VARIANTS:
        eq, trades = run_variant(prices, v)
        m = metrics(eq)
        prot = protective_pct_2022(trades)
        passed, fails = gate_pass(m, prot)
        results.append({
            "variant": v.name, "metrics": m, "prot_2022": prot,
            "passed": passed, "fails": fails, "equity": eq, "trades": trades,
        })

    print(f"\n{'='*92}")
    print(f"  COMPARISON TABLE  (initial ${INITIAL_CAPITAL:,.0f}, period varies by ETF inception)")
    print(f"{'='*92}")
    header = f"  {'Variant':<18} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'MDD date':>11} {'2022 DD':>8} {'2022 prot':>10} {'Gate':>6}"
    print(header)
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*8} {'-'*11} {'-'*8} {'-'*10} {'-'*6}")
    for r in results:
        m = r["metrics"]
        gate = "PASS" if r["passed"] else "FAIL"
        mdd_date = m["max_dd_date"].strftime("%Y-%m-%d") if pd.notna(m.get("max_dd_date")) else "n/a"
        print(
            f"  {r['variant']:<18} {m['cagr']:>7.1%} {m['sharpe']:>7.2f} "
            f"{m['max_dd']:>7.1%} {mdd_date:>11} {m['max_dd_2022']:>7.1%} "
            f"{r['prot_2022']:>9.0%} {gate:>6}"
        )
        if r["fails"]:
            for f in r["fails"]:
                print(f"    - {f}")

    # Pick best Sharpe among passing variants
    passing = [r for r in results if r["passed"]]
    print(f"\n{'='*92}")
    if passing:
        winner = max(passing, key=lambda r: r["metrics"]["sharpe"])
        print(f"  WINNER: {winner['variant']}  (best Sharpe among {len(passing)} passing variants)")
        m = winner["metrics"]
        print(f"  CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']:.1%}  "
              f"2022 DD {m['max_dd_2022']:.1%}  2022 protective {winner['prot_2022']:.0%}")
    else:
        print("  NO VARIANT PASSED ALL FOUR GATES.")
        # Show the closest-to-passing
        scored = sorted(results, key=lambda r: len(r["fails"]))
        closest = scored[0]
        print(f"  Closest: {closest['variant']} (failed: {', '.join(closest['fails'])})")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    main()
