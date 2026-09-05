"""Backtest engine for the BTC/ETH crypto momentum strategy."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.costs import (
    ALPACA_CRYPTO_TAKER_BPS,
    cost_multiplier,
    legs_for_switch,
    turnover_summary,
)
from signals.crypto_momentum import CryptoMomentumConfig, CryptoMomentumState, compute_crypto_signal

# The strategy evaluates weekly. Which weekday is a free choice with no economic
# content — and a 2026-09-05 audit found it carried the entire result (91.1% CAGR
# on Monday, 1.2% on Saturday). Never select one; run all seven and average.
# See run_crypto_tranched.
TRANCHE_WEEKDAYS = (0, 1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class CryptoBacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    summary: pd.DataFrame
    windows: pd.DataFrame
    gates: pd.DataFrame
    turnover: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.gates["passed"].all())


def fetch_crypto_prices(
    start: str = "2018-01-01", end: str = "2025-01-01", max_retries: int = 3
) -> pd.DataFrame:
    tickers = ["BTC-USD", "ETH-USD"]
    last_err: Exception | None = None
    for attempt in range(max_retries):
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            last_err = RuntimeError("No BTC/ETH data returned from yfinance")
            time.sleep(2**attempt)
            continue
        prices = raw["Close"].rename(columns={"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"})
        prices = prices.dropna(how="all").ffill()
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        incomplete = [c for c in prices.columns if prices[c].isna().sum() > len(prices) * 0.05]
        if not incomplete:
            return prices.dropna()
        last_err = RuntimeError(f"Incomplete price data for {incomplete} (attempt {attempt + 1})")
        time.sleep(2**attempt)
    raise RuntimeError(
        f"Failed to fetch complete BTC/ETH price data after {max_retries} attempts: {last_err}"
    )


def _metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = returns.std() * np.sqrt(365)
    sharpe = (returns.mean() * 365) / vol if vol > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return {
        "ending_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(sharpe),
    }


def _normalize_window(equity: pd.Series, capital: float) -> pd.Series:
    return equity / equity.iloc[0] * capital


def run_crypto_backtest(
    prices: pd.DataFrame,
    cfg: CryptoMomentumConfig,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    cost_bps: float = ALPACA_CRYPTO_TAKER_BPS,
    rebalance_weekday: int = 0,
) -> CryptoBacktestResult:
    """Backtest one weekday tranche of the crypto momentum strategy.

    Args:
        cost_bps: per-leg trading cost. Defaults to Alpaca's real taker fee — a
            zero-cost run has to be asked for explicitly.
        rebalance_weekday: 0=Monday. **Prefer run_crypto_tranched**, which averages
            all seven; a single weekday is a bet on a parameter with no economics
            behind it.
    """
    prices = prices.loc[start:end].copy()
    cash = cfg.capital
    qty = 0.0
    held: str | None = None
    state = CryptoMomentumState(peak=cfg.capital, cash_value=cfg.capital)
    min_history = max(cfg.abs_lookback + cfg.abs_skip, cfg.rel_lookback + cfg.rel_skip)
    legs = 0

    daily_values: list[tuple[pd.Timestamp, float, str]] = []
    trades: list[dict] = []

    for i, date in enumerate(prices.index):
        row = prices.iloc[i]
        portfolio_value = cash + (qty * row[held] if held else 0.0)

        if held and state.peak > 0 and portfolio_value / state.peak - 1.0 <= cfg.cb_threshold:
            cash = portfolio_value * cost_multiplier(1, cost_bps)
            legs += 1
            qty = 0.0
            held = None
            state.cash_value = cash
            state.last_target = None
            trades.append(
                {
                    "date": date,
                    "target": "USDC",
                    "regime": "circuit_breaker",
                    "btc_abs": np.nan,
                    "btc_rel": np.nan,
                    "eth_rel": np.nan,
                    "portfolio_value": portfolio_value,
                }
            )

        if date.weekday() == rebalance_weekday and i >= min_history:
            portfolio_value = cash + (qty * row[held] if held else 0.0)
            signal, new_state = compute_crypto_signal(prices.iloc[: i + 1], state, portfolio_value, cfg)
            target = signal.target

            if held != target:
                switch_legs = legs_for_switch(held, target)
                legs += switch_legs
                cash = portfolio_value * cost_multiplier(switch_legs, cost_bps)
                qty = 0.0
                held = None
                if target:
                    qty = cash / row[target]
                    cash = 0.0
                    held = target
                    new_state.cash_value = 0.0
                else:
                    new_state.cash_value = cash

            new_state.last_eval_date = date.date().isoformat()
            state = new_state
            trades.append(
                {
                    "date": date,
                    "target": target or "USDC",
                    "regime": signal.regime,
                    "btc_abs": signal.btc_abs_return,
                    "btc_rel": signal.relative_scores.get("BTC/USD", np.nan),
                    "eth_rel": signal.relative_scores.get("ETH/USD", np.nan),
                    "portfolio_value": portfolio_value,
                }
            )

        value_after = cash + (qty * row[held] if held else 0.0)
        state.peak = max(state.peak, value_after)
        daily_values.append((date, value_after, held or "USDC"))

    equity_df = pd.DataFrame(daily_values, columns=["date", "value", "holding"]).set_index("date")
    equity = equity_df["value"].rename("strategy")
    trades_df = pd.DataFrame(trades)

    summary_rows = [{"series": "Strategy", **_metrics(equity)}]
    for symbol in cfg.universe:
        bench = prices[symbol].reindex(equity.index).ffill()
        bench = bench / bench.iloc[0] * cfg.capital
        summary_rows.append({"series": f"{symbol.split('/')[0]} buy&hold", **_metrics(bench)})
    summary = pd.DataFrame(summary_rows).set_index("series")

    window_rows = []
    for label, w_start, w_end in [
        ("Train 2018-2021", "2018-01-01", "2021-12-31"),
        ("Test 2022-2024", "2022-01-01", "2024-12-31"),
    ]:
        sub = equity.loc[w_start:w_end]
        if len(sub) > 2:
            window_rows.append({"window": label, **_metrics(_normalize_window(sub, cfg.capital))})
    windows = pd.DataFrame(window_rows).set_index("window")

    usdc_2022_pct = _usdc_2022_months(trades_df)
    strategy_metrics = _metrics(equity)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    turnover = turnover_summary(legs, years, cost_bps)

    # The gate that was missing. A CAGR is meaningless without the thing it had to
    # beat: over 2021-2024 the strategy posted 91.1% and BTC posted 33.6%, and only
    # one of those numbers was ever on the CV.
    btc_cagr = float(summary.loc["BTC buy&hold", "cagr"]) if "BTC buy&hold" in summary.index else float("nan")

    gates = pd.DataFrame(
        [
            {
                "gate": "CAGR > BTC buy & hold",
                "value": strategy_metrics["cagr"] - btc_cagr,
                "passed": bool(strategy_metrics["cagr"] > btc_cagr),
            },
            {"gate": "CAGR > 40%", "value": strategy_metrics["cagr"], "passed": strategy_metrics["cagr"] > 0.40},
            {
                "gate": "Max drawdown better than -60%",
                "value": strategy_metrics["max_drawdown"],
                "passed": strategy_metrics["max_drawdown"] > -0.60,
            },
            {"gate": "USDC in >=60% of 2022 months", "value": usdc_2022_pct, "passed": usdc_2022_pct >= 0.60},
            {"gate": "Sharpe > 0.6", "value": strategy_metrics["sharpe"], "passed": strategy_metrics["sharpe"] > 0.60},
        ]
    ).set_index("gate")

    return CryptoBacktestResult(
        equity=equity, trades=trades_df, summary=summary, windows=windows, gates=gates, turnover=turnover
    )


def run_crypto_tranched(
    prices: pd.DataFrame,
    cfg: CryptoMomentumConfig,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    cost_bps: float = ALPACA_CRYPTO_TAKER_BPS,
    weekdays: tuple[int, ...] = TRANCHE_WEEKDAYS,
) -> CryptoBacktestResult:
    """Run one sub-account per weekday at 1/N capital and combine them.

    This is the honest way to run a weekly strategy. Picking a weekday is picking a
    number out of a hat and then reporting the best draw: across the seven, the
    2021-2024 out-of-sample CAGR ran 91.1% / 86.5% / 59.9% / 11.6% / 14.8% / 1.2% /
    19.3%. The average is what the strategy is actually worth, and it is directly
    implementable — seven sub-accounts, each rebalancing on its own day.

    Costs are unchanged per unit of capital: each tranche trades 1/N of the account.
    """
    if not weekdays:
        raise ValueError("weekdays must not be empty")

    runs = [
        run_crypto_backtest(prices, cfg, start=start, end=end, cost_bps=cost_bps, rebalance_weekday=wd)
        for wd in weekdays
    ]

    returns = [r.equity.pct_change().fillna(0.0) for r in runs]
    blended = sum(returns) / float(len(returns))
    equity = (cfg.capital * (1.0 + blended).cumprod()).rename("strategy")

    trades = pd.concat(
        [r.trades.assign(weekday=wd) for r, wd in zip(runs, weekdays) if not r.trades.empty],
        ignore_index=True,
    ) if any(not r.trades.empty for r in runs) else pd.DataFrame()

    summary_rows = [{"series": "Strategy (tranched)", **_metrics(equity)}]
    for symbol in cfg.universe:
        bench = prices.loc[start:end, symbol].reindex(equity.index).ffill()
        bench = bench / bench.iloc[0] * cfg.capital
        summary_rows.append({"series": f"{symbol.split('/')[0]} buy&hold", **_metrics(bench)})
    summary = pd.DataFrame(summary_rows).set_index("series")

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_legs = int(sum(r.turnover.get("legs", 0.0) for r in runs) / len(runs))
    turnover = turnover_summary(total_legs, years, cost_bps)

    m = _metrics(equity)
    btc_cagr = float(summary.loc["BTC buy&hold", "cagr"]) if "BTC buy&hold" in summary.index else float("nan")
    gates = pd.DataFrame(
        [
            {"gate": "CAGR > BTC buy & hold", "value": m["cagr"] - btc_cagr, "passed": bool(m["cagr"] > btc_cagr)},
            {"gate": "CAGR > 40%", "value": m["cagr"], "passed": m["cagr"] > 0.40},
            {
                "gate": "Max drawdown better than -60%",
                "value": m["max_drawdown"],
                "passed": m["max_drawdown"] > -0.60,
            },
            {"gate": "Sharpe > 0.6", "value": m["sharpe"], "passed": m["sharpe"] > 0.60},
        ]
    ).set_index("gate")

    windows = pd.DataFrame()
    return CryptoBacktestResult(
        equity=equity, trades=trades, summary=summary, windows=windows, gates=gates, turnover=turnover
    )


def _usdc_2022_months(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    rows = trades[(trades["date"] >= "2022-01-01") & (trades["date"] <= "2022-12-31")].copy()
    rows = rows[rows["regime"] != "circuit_breaker"]
    if rows.empty:
        return float("nan")
    rows["month"] = rows["date"].dt.to_period("M")
    monthly = rows.groupby("month").tail(1)
    return float((monthly["target"] == "USDC").mean())
