"""Backtest engine for the V4 dual-momentum leveraged ETF strategy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from signals.dual_momentum import V4Config, V4State, compute_signal


@dataclass(frozen=True)
class DualMomentumBacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    summary: pd.DataFrame
    windows: pd.DataFrame
    gates: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(self.gates["passed"].all())


def fetch_etf_prices(start: str = "2010-03-01", end: str = "2024-12-31") -> pd.DataFrame:
    tickers = ["TQQQ", "UPRO", "SOXL", "TLT", "SPY"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    prices = raw["Close"][tickers].dropna(how="all").ffill()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


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


def run_dual_momentum_backtest(
    prices: pd.DataFrame,
    cfg: V4Config,
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
) -> DualMomentumBacktestResult:
    prices = prices.loc[start:end].copy()
    cash = initial_capital
    shares = 0.0
    held: str | None = None
    state = V4State(peak=initial_capital, cash_value=initial_capital)
    min_history = cfg.abs_lookback + cfg.skip + 1

    daily_values: list[tuple[pd.Timestamp, float, str]] = []
    trades: list[dict] = []

    for i, date in enumerate(prices.index):
        row = prices.iloc[i]
        portfolio_value = cash + (shares * float(row[held]) if held else 0.0)

        # Circuit breaker — check daily
        if state.peak > 0 and held:
            dd = (state.peak - portfolio_value) / state.peak
            if dd >= cfg.cb_threshold and not state.in_cb:
                cash = portfolio_value
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
        if i >= min_history and _is_last_trading_day_of_month(prices.index, i):
            portfolio_value = cash + (shares * float(row[held]) if held else 0.0)
            signal, new_state = compute_signal(prices.iloc[: i + 1], state, portfolio_value, cfg)
            target = signal.target

            if held != target:
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

        value_after = cash + (shares * float(row[held]) if held else 0.0)
        state.peak = max(state.peak, value_after)
        daily_values.append((date, value_after, held or "CASH"))

    equity_df = pd.DataFrame(daily_values, columns=["date", "value", "holding"]).set_index("date")
    equity = equity_df["value"].rename("strategy")
    trades_df = pd.DataFrame(trades)

    # Summary vs benchmarks
    summary_rows = [{"series": "V4 Strategy", **_metrics(equity)}]
    for ticker in ["SPY", "QQQ"]:
        try:
            bench_raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            bench = bench_raw["Close"].squeeze().reindex(equity.index).ffill()
            bench = bench / bench.iloc[0] * initial_capital
            summary_rows.append({"series": f"{ticker} buy&hold", **_metrics(bench)})
        except Exception:
            pass
    summary = pd.DataFrame(summary_rows).set_index("series")

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
    gates = pd.DataFrame([
        {"gate": "CAGR > 20%", "value": m["cagr"], "passed": m["cagr"] > 0.20},
        {"gate": "Max drawdown better than -75%", "value": m["max_drawdown"], "passed": m["max_drawdown"] > -0.75},
        {"gate": "Sharpe > 0.5", "value": m["sharpe"], "passed": m["sharpe"] > 0.50},
        {"gate": "2022 drawdown better than -40%", "value": dd_2022, "passed": dd_2022 > -0.40},
    ]).set_index("gate")

    return DualMomentumBacktestResult(
        equity=equity, trades=trades_df, summary=summary, windows=windows, gates=gates
    )


def _max_dd_year(equity: pd.Series, year: int) -> float:
    sub = equity.loc[str(year)]
    if sub.empty:
        return float("nan")
    return float((sub / sub.cummax() - 1.0).min())
