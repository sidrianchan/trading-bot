"""Backtest engine for the V4 dual-momentum leveraged ETF strategy."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.costs import (
    ETF_SPREAD_SLIPPAGE_BPS,
    cost_multiplier,
    legs_for_switch,
    turnover_summary,
)
from signals.dual_momentum import V4Config, V4State, compute_signal

# Which day of the month to rebalance is a free choice with no economic content.
# A 2026-09-05 audit found it ran the result from 42.2% CAGR (month end) down to
# 7.1% (six trading days earlier) — month end ranked 1 of 11. Run several and
# average. See run_dual_momentum_tranched.
TRANCHE_OFFSETS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class DualMomentumBacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    summary: pd.DataFrame
    windows: pd.DataFrame
    gates: pd.DataFrame
    turnover: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.gates["passed"].all())


def fetch_etf_prices(
    start: str = "2010-03-01", end: str = "2024-12-31", max_retries: int = 3
) -> pd.DataFrame:
    tickers = ["TQQQ", "UPRO", "SOXL", "TLT", "SPY"]
    last_err: Exception | None = None
    for attempt in range(max_retries):
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        try:
            prices = raw["Close"][tickers].dropna(how="all").ffill()
        except KeyError as exc:
            last_err = exc
            time.sleep(2**attempt)
            continue
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        # A flaky yfinance batch response can silently return an all-NaN column for
        # one ticker instead of erroring; ffill() then leaves it NaN for the whole
        # series, which changes strategy selection without any visible failure.
        incomplete = [t for t in tickers if prices[t].isna().sum() > len(prices) * 0.05]
        if not incomplete:
            return prices
        last_err = RuntimeError(f"Incomplete price data for {incomplete} (attempt {attempt + 1})")
        time.sleep(2**attempt)
    raise RuntimeError(
        f"Failed to fetch complete ETF price data for {tickers} after {max_retries} attempts: {last_err}"
    )


def _metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return {
        "ending_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(sharpe),
    }


def _is_last_trading_day_of_month(dates: pd.DatetimeIndex, i: int) -> bool:
    """True if dates[i] is the last trading day in its calendar month."""
    if i + 1 >= len(dates):
        return True
    return dates[i].month != dates[i + 1].month


def rebalance_indices(dates: pd.DatetimeIndex, offset: int = 0) -> set[int]:
    """Positions to rebalance on: ``offset`` trading days before each month end.

    offset=0 reproduces the original month-end rule exactly.
    """
    if offset < 0:
        raise ValueError("offset must be >= 0")
    month_ends = [i for i in range(len(dates)) if _is_last_trading_day_of_month(dates, i)]
    return {i - offset for i in month_ends if i - offset >= 0}


def run_dual_momentum_backtest(
    prices: pd.DataFrame,
    cfg: V4Config,
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
    cost_bps: float = ETF_SPREAD_SLIPPAGE_BPS,
    rebalance_offset: int = 0,
) -> DualMomentumBacktestResult:
    """Backtest one rebalance-timing tranche of the V4 dual-momentum strategy.

    Args:
        cost_bps: per-leg spread + slippage. Alpaca charges no equity commission,
            but the spread is not zero and a zero-cost run must be asked for.
        rebalance_offset: trading days before month end. **Prefer
            run_dual_momentum_tranched**, which averages several offsets.
    """
    prices = prices.loc[start:end].copy()
    cash = initial_capital
    shares = 0.0
    held: str | None = None
    state = V4State(peak=initial_capital, cash_value=initial_capital)
    min_history = cfg.abs_lookback + cfg.skip + 1
    rebalance_on = rebalance_indices(prices.index, rebalance_offset)
    legs = 0

    daily_values: list[tuple[pd.Timestamp, float, str]] = []
    trades: list[dict] = []

    for i, date in enumerate(prices.index):
        row = prices.iloc[i]
        portfolio_value = cash + (shares * float(row[held]) if held else 0.0)

        # Circuit breaker — check daily
        if state.peak > 0 and held:
            dd = (state.peak - portfolio_value) / state.peak
            if dd >= cfg.cb_threshold and not state.in_cb:
                cash = portfolio_value * cost_multiplier(1, cost_bps)
                legs += 1
                shares = 0.0
                held = None
                state.in_cb = True
                state.cb_confirm_count = 0
                state.cash_value = cash
                state.last_target = None
                trades.append({
                    "date": date, "target": "CASH", "regime": "circuit_breaker",
                    "spy_ret": np.nan, "top_candidate": None, "score": np.nan,
                    "portfolio_value": portfolio_value,
                })

        # Monthly rebalance on last trading day of month
        if i >= min_history and i in rebalance_on:
            portfolio_value = cash + (shares * float(row[held]) if held else 0.0)
            signal, new_state = compute_signal(prices.iloc[: i + 1], state, portfolio_value, cfg)
            target = signal.target

            if held != target:
                switch_legs = legs_for_switch(held, target)
                legs += switch_legs
                cash = portfolio_value * cost_multiplier(switch_legs, cost_bps)
                shares = 0.0
                held = None
                if target:
                    price = float(row[target])
                    shares = cash / price
                    cash = 0.0
                    held = target
                    new_state.cash_value = 0.0
                else:
                    new_state.cash_value = cash

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

        value_after = cash + (shares * float(row[held]) if held else 0.0)
        state.peak = max(state.peak, value_after)
        daily_values.append((date, value_after, held or "CASH"))

    equity_df = pd.DataFrame(daily_values, columns=["date", "value", "holding"]).set_index("date")
    equity = equity_df["value"].rename("strategy")
    trades_df = pd.DataFrame(trades)

    summary = _benchmark_summary(equity, prices, initial_capital, label="V4 Strategy")

    # Sub-period windows
    window_rows = []
    for label, w_start, w_end in [
        ("Bull 2010-2021", "2010-03-01", "2021-12-31"),
        ("Bear 2022", "2022-01-01", "2022-12-31"),
        ("Recovery 2023-2024", "2023-01-01", "2024-12-31"),
    ]:
        sub = equity.loc[w_start:w_end]
        if len(sub) > 2:
            norm = sub / sub.iloc[0] * initial_capital
            window_rows.append({"window": label, **_metrics(norm)})
    windows = pd.DataFrame(window_rows).set_index("window")

    m = _metrics(equity)
    dd_2022 = _max_dd_year(equity, 2022)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    turnover = turnover_summary(legs, years, cost_bps)
    gates = _gates(m, dd_2022, summary)

    return DualMomentumBacktestResult(
        equity=equity, trades=trades_df, summary=summary, windows=windows, gates=gates, turnover=turnover
    )


def _benchmark_summary(
    equity: pd.Series, prices: pd.DataFrame, capital: float, label: str
) -> pd.DataFrame:
    """Strategy against the things it actually has to beat.

    Uses the already-fetched price frame rather than re-downloading, which also
    takes a network round-trip out of every walk-forward iteration. TQQQ is the
    benchmark that matters: the strategy holds it most of the time, so "beat SPY"
    is the wrong bar and flatters the result.
    """
    rows = [{"series": label, **_metrics(equity)}]
    for ticker in ("SPY", "TQQQ"):
        if ticker not in prices.columns:
            continue
        bench = prices[ticker].reindex(equity.index).ffill()
        if bench.isna().all() or float(bench.iloc[0]) <= 0:
            continue
        bench = bench / bench.iloc[0] * capital
        rows.append({"series": f"{ticker} buy&hold", **_metrics(bench)})
    return pd.DataFrame(rows).set_index("series")


def _gates(m: dict[str, float], dd_2022: float, summary: pd.DataFrame) -> pd.DataFrame:
    """Validation gates, benchmark-relative first.

    The Sharpe-vs-TQQQ gate is the one added on 2026-09-05. The strategy's measured
    contribution is drawdown control, not return: over 2016-2024 its excess return
    over simply holding TQQQ was -8.98%/yr with t = -0.45. A gate that only asks for
    "CAGR > 20%" passes on TQQQ's beta and calls it a strategy.
    """
    tqqq_sharpe = float(summary.loc["TQQQ buy&hold", "sharpe"]) if "TQQQ buy&hold" in summary.index else float("nan")
    tqqq_dd = float(summary.loc["TQQQ buy&hold", "max_drawdown"]) if "TQQQ buy&hold" in summary.index else float("nan")
    return pd.DataFrame([
        {"gate": "Sharpe > TQQQ buy & hold", "value": m["sharpe"] - tqqq_sharpe, "passed": bool(m["sharpe"] > tqqq_sharpe)},
        {"gate": "Max drawdown better than TQQQ", "value": m["max_drawdown"] - tqqq_dd, "passed": bool(m["max_drawdown"] > tqqq_dd)},
        {"gate": "CAGR > 20%", "value": m["cagr"], "passed": m["cagr"] > 0.20},
        {"gate": "Max drawdown better than -75%", "value": m["max_drawdown"], "passed": m["max_drawdown"] > -0.75},
        {"gate": "Sharpe > 0.5", "value": m["sharpe"], "passed": m["sharpe"] > 0.50},
        {"gate": "2022 drawdown better than -40%", "value": dd_2022, "passed": dd_2022 > -0.40},
    ]).set_index("gate")


def run_dual_momentum_tranched(
    prices: pd.DataFrame,
    cfg: V4Config,
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
    cost_bps: float = ETF_SPREAD_SLIPPAGE_BPS,
    offsets: tuple[int, ...] = TRANCHE_OFFSETS,
) -> DualMomentumBacktestResult:
    """Run one sub-account per rebalance offset at 1/N capital and combine them.

    Removes the timing luck that made month end look like a strategy choice rather
    than a coin flip. Implementable live as N sub-accounts rebalancing on N days.
    """
    if not offsets:
        raise ValueError("offsets must not be empty")

    runs = [
        run_dual_momentum_backtest(
            prices, cfg, start=start, end=end, initial_capital=initial_capital,
            cost_bps=cost_bps, rebalance_offset=off,
        )
        for off in offsets
    ]

    blended = sum(r.equity.pct_change().fillna(0.0) for r in runs) / float(len(runs))
    equity = (initial_capital * (1.0 + blended).cumprod()).rename("strategy")

    trades = pd.concat(
        [r.trades.assign(offset=off) for r, off in zip(runs, offsets) if not r.trades.empty],
        ignore_index=True,
    ) if any(not r.trades.empty for r in runs) else pd.DataFrame()

    summary = _benchmark_summary(equity, prices.loc[start:end], initial_capital, label="V4 (tranched)")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    legs = int(sum(r.turnover.get("legs", 0.0) for r in runs) / len(runs))
    turnover = turnover_summary(legs, years, cost_bps)
    m = _metrics(equity)
    gates = _gates(m, _max_dd_year(equity, 2022), summary)

    return DualMomentumBacktestResult(
        equity=equity, trades=trades, summary=summary, windows=pd.DataFrame(),
        gates=gates, turnover=turnover,
    )


def _max_dd_year(equity: pd.Series, year: int) -> float:
    sub = equity[equity.index.year == year]
    if sub.empty:
        return float("nan")
    return float((sub / sub.cummax() - 1.0).min())
