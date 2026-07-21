"""Cadence comparison harness for the V4 dual-momentum ETF strategy.

Phase 1 analysis ONLY — no live changes. Holds ALL V4 signal parameters fixed
and varies *only* the rebalance cadence, so any difference is attributable to
cadence, not to a change in selection logic.

Reuses the live signal path (`signals.dual_momentum.compute_signal`, the exact
function the production bot calls) and the engine helpers from
`backtester.dual_momentum`, so results faithfully mirror live behavior.

Cadences compared (same data, same params):
  1. monthly              — baseline; last trading day of each calendar month
                            (reproduces the current live behavior / reference).
  2. weekly               — last trading day of each ISO week; selection logic
                            unchanged, only evaluation frequency changes.
  3. monthly_weekly_exit  — selection still monthly, but a weekly check forces a
                            move to cash if SPY absolute momentum has turned
                            negative. Isolates the real weakness of monthly
                            (slow exits) without adding entry whipsaw.

Run:  .venv/bin/python -m backtester.cadence_comparison
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.dual_momentum import (
    _max_dd_year,
    _metrics,
    _is_last_trading_day_of_month,
    fetch_etf_prices,
)
from signals.dual_momentum import V4Config, V4State, compute_signal, total_return_skip


def _is_week_end(dates: pd.DatetimeIndex, i: int) -> bool:
    """True if dates[i] is the last trading day in its ISO week."""
    if i + 1 >= len(dates):
        return True
    a = dates[i].isocalendar()
    b = dates[i + 1].isocalendar()
    return (a.year, a.week) != (b.year, b.week)


@dataclass(frozen=True)
class CadenceResult:
    name: str
    equity: pd.Series
    trades: pd.DataFrame
    turnover: int  # number of position switches (incl. to/from CASH)


def run_cadence_backtest(
    prices: pd.DataFrame,
    cfg: V4Config,
    cadence: str,
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
) -> CadenceResult:
    """Mirror of run_dual_momentum_backtest with a parameterized rebalance trigger.

    cadence:
      "monthly"             -> rebalance on last trading day of month
      "weekly"              -> rebalance on last trading day of ISO week
      "monthly_weekly_exit" -> monthly rebalance + weekly defensive exit to cash
                               when SPY abs-momentum <= 0
    """
    if cadence not in {"monthly", "weekly", "monthly_weekly_exit"}:
        raise ValueError(f"unknown cadence: {cadence}")

    prices = prices.loc[start:end].copy()
    cash = initial_capital
    shares = 0.0
    held: str | None = None
    state = V4State(peak=initial_capital, cash_value=initial_capital)
    min_history = cfg.abs_lookback + cfg.skip + 1
    spy_col = cfg.benchmark_filter

    daily_values: list[tuple[pd.Timestamp, float, str]] = []
    trades: list[dict] = []
    turnover = 0

    for i, date in enumerate(prices.index):
        row = prices.iloc[i]
        portfolio_value = cash + (shares * float(row[held]) if held else 0.0)

        # Circuit breaker — checked daily (identical to live engine)
        if state.peak > 0 and held:
            dd = (state.peak - portfolio_value) / state.peak
            if dd >= cfg.cb_threshold and not state.in_cb:
                cash = portfolio_value
                shares = 0.0
                held = None
                turnover += 1
                state.in_cb = True
                state.cb_confirm_count = 0
                state.cash_value = cash
                state.last_target = None
                trades.append({
                    "date": date, "target": "CASH", "regime": "circuit_breaker",
                    "spy_ret": np.nan, "top_candidate": None, "score": np.nan,
                    "portfolio_value": portfolio_value,
                })

        is_month_end = _is_last_trading_day_of_month(prices.index, i)
        is_week_end = _is_week_end(prices.index, i)

        do_select = (cadence == "weekly" and is_week_end) or (
            cadence in {"monthly", "monthly_weekly_exit"} and is_month_end
        )

        if i >= min_history and do_select:
            portfolio_value = cash + (shares * float(row[held]) if held else 0.0)
            signal, new_state = compute_signal(prices.iloc[: i + 1], state, portfolio_value, cfg)
            target = signal.target

            if held != target:
                turnover += 1
                cash = portfolio_value
                shares = 0.0
                held = None
                if target:
                    price = float(row[target])
                    shares = cash / price
                    cash = 0.0
                    held = target
                    new_state.cash_value = 0.0
                else:
                    new_state.cash_value = portfolio_value

            new_state.last_eval_date = date.date().isoformat()
            state = new_state
            trades.append({
                "date": date,
                "target": target or "CASH",
                "regime": signal.regime,
                "spy_ret": signal.spy_lookback_return,
                "top_candidate": target,
                "score": signal.candidate_scores.get(target, np.nan) if target else np.nan,
                "portfolio_value": portfolio_value,
            })

        # Weekly defensive exit (monthly_weekly_exit only): on a non-rebalance
        # week-end, move to cash if SPY absolute momentum has turned negative.
        elif (
            cadence == "monthly_weekly_exit"
            and is_week_end
            and i >= min_history
            and held is not None
        ):
            spy_ret = total_return_skip(prices[spy_col].iloc[: i + 1], cfg.abs_lookback, cfg.skip)
            if not pd.isna(spy_ret) and spy_ret <= 0:
                portfolio_value = cash + shares * float(row[held])
                cash = portfolio_value
                shares = 0.0
                held = None
                turnover += 1
                state.cash_value = cash
                state.last_target = None
                trades.append({
                    "date": date, "target": "CASH", "regime": "weekly_defensive_exit",
                    "spy_ret": spy_ret, "top_candidate": None, "score": np.nan,
                    "portfolio_value": portfolio_value,
                })

        value_after = cash + (shares * float(row[held]) if held else 0.0)
        state.peak = max(state.peak, value_after)
        daily_values.append((date, value_after, held or "CASH"))

    equity_df = pd.DataFrame(daily_values, columns=["date", "value", "holding"]).set_index("date")
    equity = equity_df["value"].rename(cadence)
    trades_df = pd.DataFrame(trades)
    return CadenceResult(name=cadence, equity=equity, trades=trades_df, turnover=turnover)


def _annual_vol(equity: pd.Series) -> float:
    returns = equity.pct_change().dropna()
    return float(returns.std() * np.sqrt(252))


def _gate_report(equity: pd.Series) -> tuple[bool, dict]:
    m = _metrics(equity)
    dd_2022 = _max_dd_year(equity, 2022)
    checks = {
        "CAGR>20%": m["cagr"] > 0.20,
        "MaxDD>-75%": m["max_drawdown"] > -0.75,
        "Sharpe>0.5": m["sharpe"] > 0.50,
        "2022DD>-40%": dd_2022 > -0.40,
    }
    return all(checks.values()), checks


def compare(
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
) -> pd.DataFrame:
    cfg = V4Config()
    prices = fetch_etf_prices(start=start, end=end)

    rows = []
    for cadence in ("monthly", "weekly", "monthly_weekly_exit"):
        res = run_cadence_backtest(prices, cfg, cadence, start, end, initial_capital)
        m = _metrics(res.equity)
        passed, _ = _gate_report(res.equity)
        rows.append({
            "cadence": cadence,
            "CAGR": m["cagr"],
            "MaxDD": m["max_drawdown"],
            "MaxDD_2022": _max_dd_year(res.equity, 2022),
            "Sharpe": m["sharpe"],
            "Vol": _annual_vol(res.equity),
            "Turnover": res.turnover,
            "GatesPass": passed,
        })
    return pd.DataFrame(rows).set_index("cadence")


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    table = compare()
    print("\n=== V4 ETF cadence comparison (2010-2024, params fixed) ===\n")
    print(table.to_string())

    base = table.loc["monthly"]
    print("\n--- recommendation ---")
    improved = []
    for cadence in ("weekly", "monthly_weekly_exit"):
        r = table.loc[cadence]
        better_risk = (r["Sharpe"] > base["Sharpe"]) or (r["MaxDD"] > base["MaxDD"])
        if better_risk and r["GatesPass"]:
            improved.append(cadence)
    if improved:
        print(f"Candidate cadence change(s) that improve risk-adjusted return "
              f"without failing gates: {', '.join(improved)}")
        print("Decision is the user's — see side-by-side above (Phase 2 only if justified).")
    else:
        print("No alternative cadence beats monthly on Sharpe/MaxDD while passing all gates.")
        print("Recommendation: KEEP monthly. No live cadence change is justified by the data.")


if __name__ == "__main__":
    main()
