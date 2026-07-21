"""Dual Momentum on Leveraged ETFs — backtest.

Strategy:
  1. ABSOLUTE momentum (regime filter): SPY 12-month total return > 0 ?
     - Yes: risk-on, candidate universe = [TQQQ, UPRO, SOXL]
     - No:  risk-off, candidate universe = [TLT] if TLT 3m > 0 else cash
  2. RELATIVE momentum: rank candidates by 3-month return, hold top-N
  3. Rebalance monthly on the last trading day, lookbacks skip the most
     recent 21 trading days to avoid short-term reversal noise.
  4. 50% drawdown circuit breaker: liquidate to cash if portfolio falls
     >= 50% from peak; resume signal evaluation on next rebalance.

Backtest:
  - Universe data: yfinance auto-adjusted close (total-return basis)
  - Period: 2010-02-09 (TQQQ inception) → 2024-12-31
  - Walk-forward: 2010-2019 in-sample, 2020-2024 out-of-sample
  - Benchmarks: SPY, QQQ
  - Initial capital: $100,000

WARNINGS (per spec — these apply to both backtest interpretation
and any future live trading):
  # WARNING: TQQQ/UPRO/SOXL can lose 90%+ in bear markets
  # WARNING: This strategy is for small capital + high risk tolerance ONLY
  # WARNING: 3x leverage decays in choppy markets (volatility drag)
  # WARNING: Past leveraged ETF performance does not predict future returns
  # WARNING: Paper trade for minimum 30 days before risking real money
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

RISK_ON = ["TQQQ", "UPRO", "SOXL"]
RISK_OFF_CANDIDATES = ["TLT"]  # rotates between TLT and cash
BENCHMARK_FILTER = "SPY"
BENCHMARKS = ["SPY", "QQQ"]
ABS_LOOKBACK = 252  # 12 months
REL_LOOKBACK = 63   # 3 months
SKIP = 21           # skip most-recent 1 month
TOP_N = 1
MAX_DD = 0.50
INITIAL_CAPITAL = 100_000.0
START = "2010-02-09"  # TQQQ inception
END = "2024-12-31"


def total_return_skip(series: pd.Series, lookback: int, skip: int) -> float:
    """Return over `lookback` trading days, ending `skip` days before today."""
    if len(series) < lookback + skip + 1:
        return float("nan")
    end = series.iloc[-(skip + 1)]
    start = series.iloc[-(lookback + skip + 1)]
    if start <= 0:
        return float("nan")
    return end / start - 1.0


def fetch_prices() -> pd.DataFrame:
    tickers = sorted(set(RISK_ON + RISK_OFF_CANDIDATES + BENCHMARKS))
    print(f"Fetching {tickers} from {START} to {END}...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all").ffill()
    print(f"  → {len(prices)} trading days, columns: {list(prices.columns)}")
    return prices


def last_trading_days(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    df = pd.DataFrame({"date": index})
    df["ym"] = df["date"].dt.to_period("M")
    return set(df.groupby("ym")["date"].max())


@dataclass
class _Trade:
    date: pd.Timestamp
    holding: str           # ETF ticker or "CASH"
    portfolio_value: float
    spy_12m_return: float
    regime: str            # "risk_on" / "risk_off" / "circuit_breaker"


def run_backtest(prices: pd.DataFrame) -> tuple[pd.Series, list[_Trade]]:
    rebalance_dates = last_trading_days(prices.index)
    cash = INITIAL_CAPITAL
    held_etf: str | None = None
    held_shares: float = 0.0
    peak = INITIAL_CAPITAL
    in_circuit_breaker = False

    daily_values: dict[pd.Timestamp, float] = {}
    trades: list[_Trade] = []

    for d in prices.index:
        row = prices.loc[d]
        equity = held_shares * row.get(held_etf, 0.0) if held_etf else 0.0
        portfolio_value = cash + equity
        peak = max(peak, portfolio_value)
        dd = (peak - portfolio_value) / peak if peak > 0 else 0.0

        # 50% drawdown circuit breaker — liquidate immediately
        if dd >= MAX_DD and held_etf is not None:
            cash = portfolio_value
            held_etf = None
            held_shares = 0.0
            in_circuit_breaker = True
            trades.append(_Trade(d, "CASH", portfolio_value, float("nan"), "circuit_breaker"))

        if d in rebalance_dates:
            history = prices.loc[:d]
            if len(history) >= ABS_LOOKBACK + SKIP + 1:
                spy_12m = total_return_skip(history[BENCHMARK_FILTER], ABS_LOOKBACK, SKIP)
                # Determine candidate universe
                if spy_12m > 0:
                    candidates = RISK_ON
                    regime = "risk_on"
                else:
                    tlt_3m = total_return_skip(history["TLT"], REL_LOOKBACK, SKIP)
                    candidates = ["TLT"] if tlt_3m > 0 else []
                    regime = "risk_off"

                # Pick top by relative momentum
                if candidates:
                    rel = {
                        etf: total_return_skip(history[etf], REL_LOOKBACK, SKIP)
                        for etf in candidates
                    }
                    rel = {k: v for k, v in rel.items() if pd.notna(v)}
                    ranked = sorted(rel.items(), key=lambda kv: -kv[1])
                    target = ranked[0][0] if ranked else None
                else:
                    target = None  # cash

                # If circuit-breaker is on, only re-enter when regime is risk_on
                # (matches user spec: "Re-enter leveraged ETFs only when SPY 12-month
                # return turns positive again"). When risk_on resumes, lift CB.
                if in_circuit_breaker and regime != "risk_on":
                    target = None
                elif in_circuit_breaker and regime == "risk_on":
                    in_circuit_breaker = False
                    peak = portfolio_value  # reset peak so CB doesn't immediately retrigger

                # Liquidate current holding if changing
                if held_etf != target:
                    if held_etf is not None:
                        cash += held_shares * row[held_etf]
                        held_shares = 0.0
                    if target is not None:
                        target_price = row[target]
                        if target_price > 0:
                            held_shares = cash / target_price
                            cash = 0.0
                            held_etf = target
                    else:
                        held_etf = None

                trades.append(_Trade(
                    d,
                    held_etf if held_etf else "CASH",
                    portfolio_value,
                    spy_12m,
                    regime,
                ))

        equity_after = held_shares * row.get(held_etf, 0.0) if held_etf else 0.0
        daily_values[d] = cash + equity_after

    return pd.Series(daily_values, name="strategy"), trades


def metrics(equity: pd.Series, label: str) -> dict:
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else float("nan")
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = dd.min()
    return {
        "label": label,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "years": years,
    }


def yearly_breakdown(strategy: pd.Series, benchmarks: dict[str, pd.Series]) -> str:
    out = ["", f"{'Year':<6} {'Strategy':>10} " + " ".join(f"{n:>10}" for n in benchmarks)]
    out.append("-" * (18 + 11 * len(benchmarks)))
    series_by_year = {}
    for series_name, series in [("strategy", strategy)] + list(benchmarks.items()):
        series_by_year[series_name] = series.groupby(series.index.year)
    years = sorted(set(strategy.index.year))
    for y in years:
        s_grp = series_by_year["strategy"].get_group(y)
        s_ret = s_grp.iloc[-1] / s_grp.iloc[0] - 1
        bench_rets = []
        for bn in benchmarks:
            b_grp = series_by_year[bn].get_group(y)
            b_ret = b_grp.iloc[-1] / b_grp.iloc[0] - 1
            bench_rets.append(b_ret)
        bench_str = " ".join(f"{r:>9.1%}" for r in bench_rets)
        out.append(f"{y:<6} {s_ret:>9.1%} {bench_str}")
    return "\n".join(out)


def regime_summary(trades: list[_Trade]) -> str:
    df = pd.DataFrame([t.__dict__ for t in trades])
    df = df[df["regime"] != "circuit_breaker"]
    counts = df["regime"].value_counts()
    holding_counts = df["holding"].value_counts()
    out = ["\nRebalance regime counts:"]
    for regime, n in counts.items():
        out.append(f"  {regime:<14} {n:>4} months")
    out.append("\nMonths held by instrument:")
    for inst, n in holding_counts.items():
        out.append(f"  {inst:<8} {n:>4} months")
    return "\n".join(out)


def detail_2022(trades: list[_Trade]) -> str:
    rows = [t for t in trades if t.date.year == 2022]
    if not rows:
        return "\n2022: no rebalance dates found"
    out = ["", "2022 month-by-month rebalance log:",
           f"{'Date':<12} {'Held':<8} {'Regime':<18} {'SPY 12m':>10} {'Portfolio':>12}"]
    out.append("-" * 64)
    for t in rows:
        spy = f"{t.spy_12m_return:>+9.1%}" if pd.notna(t.spy_12m_return) else "      n/a"
        out.append(f"{t.date.date()!s:<12} {t.holding:<8} {t.regime:<18} {spy:>10} ${t.portfolio_value:>10,.0f}")
    return "\n".join(out)


def gate_check(m: dict, in_2022_protective_pct: float) -> tuple[bool, list[str]]:
    failures = []
    if m["cagr"] <= 0.20:
        failures.append(f"CAGR {m['cagr']:.1%} <= 20% gate")
    if m["max_dd"] <= -0.60:
        failures.append(f"Max DD {m['max_dd']:.1%} worse than -60% gate")
    if m["sharpe"] <= 0.50:
        failures.append(f"Sharpe {m['sharpe']:.2f} <= 0.50 gate")
    if in_2022_protective_pct < 0.50:
        failures.append(f"In cash/TLT {in_2022_protective_pct:.0%} of 2022, < 50% gate")
    return (len(failures) == 0, failures)


def main() -> None:
    prices = fetch_prices()
    strategy_eq, trades = run_backtest(prices)

    # Re-base benchmarks to same start date as strategy first non-NaN value
    bench_eq = {}
    for bn in BENCHMARKS:
        b = prices[bn].reindex(strategy_eq.index).ffill()
        bench_eq[bn] = b / b.iloc[0] * INITIAL_CAPITAL

    print(f"\n{'='*78}\n  DUAL MOMENTUM LEVERAGED ETF BACKTEST\n{'='*78}")
    print(f"  Period: {strategy_eq.index[0].date()} → {strategy_eq.index[-1].date()}")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")

    rows = [
        metrics(strategy_eq, "Strategy"),
        *(metrics(bench_eq[bn], bn) for bn in BENCHMARKS),
    ]
    print(f"\n  {'Metric':<14} " + " ".join(f"{r['label']:>12}" for r in rows))
    print(f"  {'-'*14} " + " ".join(f"{'-'*12}" for _ in rows))
    print(f"  {'CAGR':<14} " + " ".join(f"{r['cagr']:>11.1%}" for r in rows))
    print(f"  {'Total Return':<14} " + " ".join(f"{r['total_return']:>11.1%}" for r in rows))
    print(f"  {'Vol (ann)':<14} " + " ".join(f"{r['vol']:>11.1%}" for r in rows))
    print(f"  {'Sharpe':<14} " + " ".join(f"{r['sharpe']:>11.2f}" for r in rows))
    print(f"  {'Max DD':<14} " + " ".join(f"{r['max_dd']:>11.1%}" for r in rows))

    print(yearly_breakdown(strategy_eq, bench_eq))

    # Best / worst calendar year for the strategy
    by_year = strategy_eq.groupby(strategy_eq.index.year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    print(f"\nStrategy best year: {by_year.idxmax()} ({by_year.max():+.1%})")
    print(f"Strategy worst year: {by_year.idxmin()} ({by_year.min():+.1%})")

    print(regime_summary(trades))
    print(detail_2022(trades))

    # 2022 protective months for gate check
    rows_2022 = [t for t in trades if t.date.year == 2022 and t.regime != "circuit_breaker"]
    protective_2022 = sum(1 for t in rows_2022 if t.holding in ("TLT", "CASH"))
    pct_2022 = protective_2022 / len(rows_2022) if rows_2022 else 0.0

    # Walk-forward split
    in_sample = strategy_eq[strategy_eq.index.year <= 2019]
    oos_sample = strategy_eq[strategy_eq.index.year >= 2020]
    print(f"\n{'─'*78}\nWalk-forward split:")
    if len(in_sample) > 1:
        m_in = metrics(in_sample, "in-sample")
        print(f"  In-sample (2010-2019):  CAGR {m_in['cagr']:.1%}  Sharpe {m_in['sharpe']:.2f}  MaxDD {m_in['max_dd']:.1%}")
    if len(oos_sample) > 1:
        m_oos = metrics(oos_sample, "out-of-sample")
        print(f"  Out-of-sample (2020+):  CAGR {m_oos['cagr']:.1%}  Sharpe {m_oos['sharpe']:.2f}  MaxDD {m_oos['max_dd']:.1%}")

    overall = metrics(strategy_eq, "Strategy")
    passed, fails = gate_check(overall, pct_2022)
    print(f"\n{'='*78}")
    print(f"  GATE CHECK")
    print(f"{'='*78}")
    print(f"  CAGR > 20%       : {'PASS' if overall['cagr'] > 0.20 else 'FAIL'} ({overall['cagr']:.1%})")
    print(f"  Max DD > -60%    : {'PASS' if overall['max_dd'] > -0.60 else 'FAIL'} ({overall['max_dd']:.1%})")
    print(f"  Sharpe > 0.50    : {'PASS' if overall['sharpe'] > 0.50 else 'FAIL'} ({overall['sharpe']:.2f})")
    print(f"  2022 protective  : {'PASS' if pct_2022 >= 0.50 else 'FAIL'} ({pct_2022:.0%} of 2022 in TLT/CASH)")
    print(f"  {'─'*40}")
    print(f"  OVERALL          : {'GATE PASSED' if passed else 'GATE FAILED'}")
    if not passed:
        for f in fails:
            print(f"     • {f}")
    print(f"{'='*78}\n")

    # Dump equity curve for further analysis
    import os
    os.makedirs("/tmp", exist_ok=True)
    pd.DataFrame({
        "strategy": strategy_eq,
        **{bn: bench_eq[bn] for bn in BENCHMARKS},
    }).to_csv("/tmp/dual_momentum_equity.csv")
    print("Equity curve saved to /tmp/dual_momentum_equity.csv")


if __name__ == "__main__":
    main()
