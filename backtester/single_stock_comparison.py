"""Single-stock short-term TA: momentum rotation vs swing trend-following.

Analysis only. Compares two short-term TA methods on a data-selected basket of
high-volatility, high-liquidity US stocks (see scripts/stock_screener.py),
against an equal-weight buy & hold baseline.

Methods:
  A. momentum_rotation — rank the basket by 3-month momentum, hold the top-K
     names equal-weight (cash if a slot's momentum is negative), rebalance
     weekly or monthly. Generalizes the live ETF dual-momentum approach.
  B. swing_trend — per stock, go long when it is in a daily uptrend
     (close>SMA200 AND EMA20>EMA50 AND ADX>=20), exit on trend break or an ATR
     trailing stop; equal-weight the sleeves. Reuses signals/indicators.

Both evaluated over the SAME window (after the longest warmup) for fairness.
Reuses total_return_skip, sma/ema/adx/atr, _metrics, _max_dd_year, _is_week_end.

CAVEAT: these names only have ~4y of history (since 2022) — one bear (2022) and
a long bull (2023-2025). Conclusions are necessarily lower-confidence than the
2010-2024 ETF study; attribution + buy&hold baseline guard against overfitting.

Run: .venv/bin/python -m backtester.single_stock_comparison
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.cadence_comparison import _is_week_end
from backtester.dual_momentum import _max_dd_year, _metrics, _is_last_trading_day_of_month
from signals.dual_momentum import total_return_skip
from signals.indicators.trend import adx, ema, sma
from signals.indicators.volatility import atr

BASKET = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA", "AMD"]
START, END = "2015-01-01", "2025-12-31"
WARMUP = 200  # longest indicator lookback (SMA200) — common fair-start anchor


@dataclass(frozen=True)
class StratResult:
    name: str
    equity: pd.Series
    turnover: int


def fetch_basket(tickers=BASKET, start=START, end=END) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = pd.DataFrame({
            "high": raw["High"][t], "low": raw["Low"][t], "close": raw["Close"][t],
        }).dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[t] = df
    return out


def _common_index(bars: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = None
    for df in bars.values():
        idx = df.index if idx is None else idx.union(df.index)
    return idx


def buy_hold(bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex, initial=100_000.0) -> StratResult:
    """Equal-weight, buy at start, never rebalance."""
    closes = pd.DataFrame({t: bars[t]["close"] for t in bars}).reindex(index).ffill()
    start_row = closes.iloc[0]
    shares = (initial / len(closes.columns)) / start_row
    equity = (closes * shares).sum(axis=1).rename("buy_hold")
    return StratResult("buy_hold", equity, turnover=len(closes.columns))


def momentum_rotation(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex,
    top_k=2, lookback=63, skip=10, cadence="monthly", initial=100_000.0,
) -> StratResult:
    closes = pd.DataFrame({t: bars[t]["close"] for t in bars}).reindex(index).ffill()
    cash = initial
    holdings: dict[str, float] = {}  # ticker -> shares
    turnover = 0
    values = []

    for i, date in enumerate(index):
        row = closes.iloc[i]
        pv = cash + sum(sh * float(row[t]) for t, sh in holdings.items())

        is_reb = (cadence == "weekly" and _is_week_end(index, i)) or (
            cadence == "monthly" and _is_last_trading_day_of_month(index, i)
        )
        if i >= WARMUP and is_reb:
            mom = {}
            for t in closes.columns:
                m = total_return_skip(closes[t].iloc[: i + 1], lookback, skip)
                if not pd.isna(m):
                    mom[t] = m
            winners = [t for t, m in sorted(mom.items(), key=lambda kv: kv[1], reverse=True) if m > 0][:top_k]

            new_set = set(winners)
            if new_set != set(holdings.keys()):
                turnover += 1
            cash = pv
            holdings = {}
            if winners:
                alloc = cash / len(winners)
                for t in winners:
                    holdings[t] = alloc / float(row[t])
                cash = 0.0

        pv_after = cash + sum(sh * float(row[t]) for t, sh in holdings.items())
        values.append((date, pv_after))

    equity = pd.DataFrame(values, columns=["date", "value"]).set_index("date")["value"].rename(f"mom_{cadence}")
    return StratResult(f"mom_{cadence}", equity, turnover)


def swing_trend(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex,
    adx_min=20.0, atr_mult=3.0, initial=100_000.0,
) -> StratResult:
    """Per-stock daily trend-following sleeves, equal-weighted."""
    n = len(bars)
    sleeve_alloc = initial / n
    sleeve_equities = []
    turnover = 0

    for t, df in bars.items():
        c = df["close"]
        s200 = sma(c, 200)
        e20, e50 = ema(c, 20), ema(c, 50)
        adx_s = adx(df, 14)
        atr_s = atr(df, 14)

        sleeve_val = sleeve_alloc
        shares = 0.0
        in_pos = False
        entry_peak = 0.0
        series = {}

        for i, date in enumerate(df.index):
            price = float(c.iloc[i])
            if in_pos:
                entry_peak = max(entry_peak, price)
                sleeve_val = shares * price
                a = atr_s.iloc[i]
                trail = entry_peak - (atr_mult * float(a) if not pd.isna(a) else 0.0)
                below_ma = (not pd.isna(s200.iloc[i])) and price < float(s200.iloc[i])
                if below_ma or price < trail:
                    sleeve_val = shares * price
                    shares = 0.0
                    in_pos = False
                    turnover += 1
            else:
                if i >= WARMUP:
                    up = (
                        (not pd.isna(s200.iloc[i])) and price > float(s200.iloc[i])
                        and float(e20.iloc[i]) > float(e50.iloc[i])
                        and (not pd.isna(adx_s.iloc[i])) and float(adx_s.iloc[i]) >= adx_min
                    )
                    if up:
                        shares = sleeve_val / price
                        in_pos = True
                        entry_peak = price
                        turnover += 1
            series[date] = shares * price if in_pos else sleeve_val
        sleeve_equities.append(pd.Series(series).reindex(index).ffill())

    equity = pd.concat(sleeve_equities, axis=1).sum(axis=1).rename("swing")
    return StratResult("swing", equity, turnover)


def _row(res: StratResult) -> dict:
    m = _metrics(res.equity)
    return {
        "strategy": res.name,
        "CAGR": m["cagr"],
        "MaxDD": m["max_drawdown"],
        "Sharpe": m["sharpe"],
        "Vol": float(res.equity.pct_change().dropna().std() * np.sqrt(252)),
        "Turnover": res.turnover,
    }


def attribution(baseline: pd.Series, variant: pd.Series, name: str) -> str:
    yb = baseline.resample("YE").last().pct_change().dropna()
    yv = variant.resample("YE").last().pct_change().dropna()
    cmp = pd.DataFrame({"buy_hold": yb.values, name: yv.values}, index=yb.index.year)
    cmp["diff"] = cmp[name] - cmp["buy_hold"]
    return f"\n--- {name} vs buy&hold (annual) ---\n" + cmp.to_string(float_format=lambda v: f"{v:+.3f}")


def compare():
    bars = fetch_basket()
    full_idx = _common_index(bars)
    eval_idx = full_idx[WARMUP:]  # fair common evaluation window

    results = [
        buy_hold(bars, eval_idx),
        momentum_rotation(bars, eval_idx, cadence="monthly"),
        momentum_rotation(bars, eval_idx, cadence="weekly"),
        swing_trend(bars, eval_idx),
    ]
    table = pd.DataFrame([_row(r) for r in results]).set_index("strategy")
    return table, {r.name: r for r in results}, eval_idx


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    table, results, idx = compare()
    print(f"\n=== Single-stock high-vol basket: method comparison ===")
    print(f"Basket: {BASKET}")
    print(f"Eval window: {idx[0].date()} -> {idx[-1].date()} ({len(idx)} trading days)\n")
    print(table.to_string())

    bh = results["buy_hold"].equity
    for name in ["mom_monthly", "mom_weekly", "swing"]:
        print(attribution(bh, results[name].equity, name))

    print("\n--- verdict ---")
    bh_sharpe = table.loc["buy_hold", "Sharpe"]
    bh_dd = table.loc["buy_hold", "MaxDD"]
    best = table["Sharpe"].idxmax()
    print(f"Highest Sharpe: {best} ({table.loc[best,'Sharpe']:.3f}). "
          f"buy&hold Sharpe={bh_sharpe:.3f}, MaxDD={bh_dd:.3f}.")
    timing_helps = [n for n in ["mom_monthly", "mom_weekly", "swing"]
                    if table.loc[n, "Sharpe"] > bh_sharpe and table.loc[n, "MaxDD"] > bh_dd]
    if timing_helps:
        print(f"Timing methods that beat buy&hold on BOTH Sharpe and MaxDD: {timing_helps}")
    else:
        print("No timing method beats buy&hold on both Sharpe and MaxDD — "
              "for these names, holding may dominate. See attribution.")


if __name__ == "__main__":
    main()
