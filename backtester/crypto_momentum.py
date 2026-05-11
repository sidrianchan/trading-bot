"""Backtest engine for the BTC/ETH crypto momentum strategy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from signals.crypto_momentum import CryptoMomentumConfig, CryptoMomentumState, compute_crypto_signal


@dataclass(frozen=True)
class CryptoBacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    summary: pd.DataFrame
    windows: pd.DataFrame
    gates: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(self.gates["passed"].all())


def fetch_crypto_prices(start: str = "2018-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    raw = yf.download(
        ["BTC-USD", "ETH-USD"],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("No BTC/ETH data returned from yfinance")
    prices = raw["Close"].rename(columns={"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"})
    prices = prices.dropna(how="all").ffill().dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


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
) -> CryptoBacktestResult:
    prices = prices.loc[start:end].copy()
    cash = cfg.capital
    qty = 0.0
    held: str | None = None
    state = CryptoMomentumState(peak=cfg.capital, cash_value=cfg.capital)
    min_history = max(cfg.abs_lookback + cfg.abs_skip, cfg.rel_lookback + cfg.rel_skip)

    daily_values: list[tuple[pd.Timestamp, float, str]] = []
    trades: list[dict] = []

    for i, date in enumerate(prices.index):
        row = prices.iloc[i]
        portfolio_value = cash + (qty * row[held] if held else 0.0)

        if held and state.peak > 0 and portfolio_value / state.peak - 1.0 <= cfg.cb_threshold:
            cash = portfolio_value
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

        if date.weekday() == 0 and i >= min_history:
            portfolio_value = cash + (qty * row[held] if held else 0.0)
            signal, new_state = compute_crypto_signal(prices.iloc[: i + 1], state, portfolio_value, cfg)
            target = signal.target

            if held != target:
                cash = portfolio_value
                qty = 0.0
                held = None
                if target:
                    qty = cash / row[target]
                    cash = 0.0
                    held = target
                    new_state.cash_value = 0.0
                else:
                    new_state.cash_value = portfolio_value

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
    gates = pd.DataFrame(
        [
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

    return CryptoBacktestResult(equity=equity, trades=trades_df, summary=summary, windows=windows, gates=gates)


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
