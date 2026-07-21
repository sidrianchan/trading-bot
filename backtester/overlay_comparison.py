"""Short-term TA risk-overlay comparison for the V4 dual-momentum ETF strategy.

Phase-1 analysis ONLY — no live changes. Holds ALL V4 *selection* parameters
fixed and compares short-term technical-analysis risk overlays head-to-head
against the current monthly baseline, so any difference is attributable to the
overlay, not to a change in selection logic.

Lesson baked in (from the rejected weekly-defensive-exit study): a flattering
summary table is not enough. For any variant that beats the baseline, run
`attribution()` to check the edge is spread across many years/regimes and is not
concentrated in 2-3 lucky events.

Variants (same data 2010-2024, same V4 params):
  1. baseline    — current live V4 (correctness anchor; must reproduce the
                   reference engine exactly).
  2. sma200      — overlay: exit the held leveraged ETF to cash when it closes
                   below its own 200-day SMA; block a monthly entry whose target
                   is below its 200-SMA. Re-entry happens on the normal monthly
                   rebalance. The most research-validated leveraged-ETF tool.
  3. adx20 / adx25 — overlay: only hold while SPY ADX(14) >= threshold; exit to
                   cash and block entries in choppy (low-ADX) markets, where
                   leveraged decay bites.
  4. accel       — *selection change*, not an overlay: replace the single 6-mo
                   SPY absolute-momentum gate with a blended 1/3/6-month
                   momentum (Accelerating Dual Momentum), evaluated weekly.
                   Reimplemented here (documented) since it diverges from the
                   live compute_signal; treat its numbers as indicative.

Run:  .venv/bin/python -m backtester.overlay_comparison
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.cadence_comparison import _is_week_end
from backtester.dual_momentum import (
    _max_dd_year,
    _metrics,
    _is_last_trading_day_of_month,
    fetch_etf_prices,
)
from signals.dual_momentum import V4Config, V4State, compute_signal, total_return_skip
from signals.indicators.trend import adx, sma


@dataclass(frozen=True)
class OverlayResult:
    name: str
    equity: pd.Series
    trades: pd.DataFrame
    turnover: int


def _fetch_spy_ohlc(start: str, end: str) -> pd.DataFrame:
    """SPY OHLC with lowercase columns, for ADX (which needs high/low/close)."""
    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False, threads=False)
    df = raw[["High", "Low", "Close"]].copy()
    df.columns = ["high", "low", "close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def run_overlay_backtest(
    prices: pd.DataFrame,
    cfg: V4Config,
    overlay: str,
    spy_adx: pd.Series | None = None,
    adx_threshold: float = 20.0,
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
) -> OverlayResult:
    """Mirror of run_dual_momentum_backtest with a daily risk overlay.

    overlay:
      "baseline" -> no overlay (must reproduce the reference engine exactly)
      "sma200"   -> 200-SMA filter on the held / candidate leveraged ETF
      "adx"      -> SPY ADX(14) >= adx_threshold gate
    Selection always uses the live compute_signal — zero drift.
    """
    if overlay not in {"baseline", "sma200", "adx"}:
        raise ValueError(f"unknown overlay: {overlay}")

    prices = prices.loc[start:end].copy()
    cash = initial_capital
    shares = 0.0
    held: str | None = None
    state = V4State(peak=initial_capital, cash_value=initial_capital)
    min_history = cfg.abs_lookback + cfg.skip + 1

    # Precompute overlay signals (all backward-looking — rolling/ewm use only
    # data up to and including each bar, so reading .loc[date] has no lookahead).
    sma200 = prices.rolling(window=200, min_periods=200).mean() if overlay == "sma200" else None

    daily_values: list[tuple[pd.Timestamp, float, str]] = []
    trades: list[dict] = []
    turnover = 0

    def blocks_entry(target: str, date: pd.Timestamp) -> bool:
        if overlay == "sma200":
            ma = sma200.loc[date, target]
            return (not pd.isna(ma)) and float(prices.loc[date, target]) < float(ma)
        if overlay == "adx":
            a = spy_adx.get(date, np.nan)
            return (not pd.isna(a)) and float(a) < adx_threshold
        return False

    def forces_exit(held_ticker: str, date: pd.Timestamp) -> bool:
        if overlay == "sma200":
            ma = sma200.loc[date, held_ticker]
            return (not pd.isna(ma)) and float(prices.loc[date, held_ticker]) < float(ma)
        if overlay == "adx":
            a = spy_adx.get(date, np.nan)
            return (not pd.isna(a)) and float(a) < adx_threshold
        return False

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

        # Daily overlay exit (between monthly rebalances)
        if overlay != "baseline" and held is not None and i >= min_history and forces_exit(held, date):
            portfolio_value = cash + shares * float(row[held])
            cash = portfolio_value
            shares = 0.0
            held = None
            turnover += 1
            state.cash_value = cash
            state.last_target = None
            trades.append({
                "date": date, "target": "CASH", "regime": f"overlay_exit_{overlay}",
                "spy_ret": np.nan, "top_candidate": None, "score": np.nan,
                "portfolio_value": portfolio_value,
            })

        # Monthly rebalance on last trading day of month (live selection logic)
        if i >= min_history and _is_last_trading_day_of_month(prices.index, i):
            portfolio_value = cash + (shares * float(row[held]) if held else 0.0)
            signal, new_state = compute_signal(prices.iloc[: i + 1], state, portfolio_value, cfg)
            target = signal.target

            # Overlay gate on entry: don't enter a target the overlay rejects.
            if target and overlay != "baseline" and blocks_entry(target, date):
                target = None

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
                "top_candidate": signal.target,
                "score": signal.candidate_scores.get(signal.target, np.nan) if signal.target else np.nan,
                "portfolio_value": portfolio_value,
            })

        value_after = cash + (shares * float(row[held]) if held else 0.0)
        state.peak = max(state.peak, value_after)
        daily_values.append((date, value_after, held or "CASH"))

    equity = (
        pd.DataFrame(daily_values, columns=["date", "value", "holding"])
        .set_index("date")["value"]
        .rename(overlay)
    )
    return OverlayResult(name=overlay, equity=equity, trades=pd.DataFrame(trades), turnover=turnover)


def run_accel_backtest(
    prices: pd.DataFrame,
    cfg: V4Config,
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
) -> OverlayResult:
    """Accelerating Dual Momentum: blended 1/3/6-mo momentum, weekly rebalance.

    NOTE: this diverges from the live compute_signal (different gate + cadence +
    immediate re-entry), so it is reimplemented here. Treat results as indicative
    of the *approach*, not a drop-in for the live strategy.
    """
    prices = prices.loc[start:end].copy()
    spy_col = cfg.benchmark_filter
    lookbacks = (21, 63, 126)  # 1, 3, 6 months
    cash = initial_capital
    shares = 0.0
    held: str | None = None
    peak = initial_capital
    in_cb = False
    min_history = max(lookbacks) + cfg.skip + 1

    daily_values: list[tuple[pd.Timestamp, float, str]] = []
    trades: list[dict] = []
    turnover = 0

    def blended(series: pd.Series) -> float:
        vals = [total_return_skip(series, lb, cfg.skip) for lb in lookbacks]
        vals = [v for v in vals if not pd.isna(v)]
        return float(np.mean(vals)) if vals else float("nan")

    for i, date in enumerate(prices.index):
        row = prices.iloc[i]
        portfolio_value = cash + (shares * float(row[held]) if held else 0.0)
        peak = max(peak, portfolio_value)

        if peak > 0 and held:
            dd = (peak - portfolio_value) / peak
            if dd >= cfg.cb_threshold and not in_cb:
                cash = portfolio_value
                shares = 0.0
                held = None
                turnover += 1
                in_cb = True
                trades.append({"date": date, "target": "CASH", "regime": "circuit_breaker",
                               "spy_ret": np.nan, "top_candidate": None, "score": np.nan,
                               "portfolio_value": portfolio_value})

        if i >= min_history and _is_week_end(prices.index, i):
            hist = prices.iloc[: i + 1]
            spy_blend = blended(hist[spy_col])
            if not pd.isna(spy_blend):
                if spy_blend > 0:  # risk-on: pick best blended among leveraged ETFs
                    in_cb = False
                    scores = {t: blended(hist[t]) for t in cfg.risk_on if pd.notna(hist[t].iloc[-1])}
                    scores = {t: s for t, s in scores.items() if not pd.isna(s)}
                    target = max(scores, key=scores.get) if scores else None
                else:  # risk-off: hold the bond candidate if it has positive blend, else cash
                    target = cfg.risk_off_candidates[0] if cfg.risk_off_candidates else None
                    if target is not None and not (blended(hist[target]) > 0):
                        target = None

                if held != target:
                    turnover += 1
                    cash = portfolio_value
                    shares = 0.0
                    held = None
                    if target:
                        shares = cash / float(row[target])
                        cash = 0.0
                        held = target
                    trades.append({"date": date, "target": target or "CASH", "regime": "accel",
                                   "spy_ret": spy_blend, "top_candidate": target, "score": np.nan,
                                   "portfolio_value": portfolio_value})

        value_after = cash + (shares * float(row[held]) if held else 0.0)
        peak = max(peak, value_after)
        daily_values.append((date, value_after, held or "CASH"))

    equity = (
        pd.DataFrame(daily_values, columns=["date", "value", "holding"])
        .set_index("date")["value"]
        .rename("accel")
    )
    return OverlayResult(name="accel", equity=equity, trades=pd.DataFrame(trades), turnover=turnover)


def _annual_vol(equity: pd.Series) -> float:
    return float(equity.pct_change().dropna().std() * np.sqrt(252))


def _gates_pass(equity: pd.Series) -> bool:
    m = _metrics(equity)
    return bool(
        m["cagr"] > 0.20
        and m["max_drawdown"] > -0.75
        and m["sharpe"] > 0.50
        and _max_dd_year(equity, 2022) > -0.40
    )


def _row(name: str, res: OverlayResult) -> dict:
    m = _metrics(res.equity)
    return {
        "variant": name,
        "CAGR": m["cagr"],
        "MaxDD": m["max_drawdown"],
        "MaxDD_2022": _max_dd_year(res.equity, 2022),
        "Sharpe": m["sharpe"],
        "Vol": _annual_vol(res.equity),
        "Turnover": res.turnover,
        "GatesPass": _gates_pass(res.equity),
    }


def attribution(baseline: pd.Series, variant: pd.Series, name: str) -> str:
    """Per-year diff + concentration check — the overfit detector."""
    def yearly(eq: pd.Series) -> pd.Series:
        return eq.resample("YE").last().pct_change().dropna()

    yb, yv = yearly(baseline), yearly(variant)
    idx = yb.index.year
    cmp = pd.DataFrame({"baseline": yb.values, name: yv.values}, index=idx)
    cmp["diff"] = cmp[name] - cmp["baseline"]
    total = cmp["diff"].sum()
    top2 = cmp["diff"].sort_values(ascending=False).head(2)
    helped = int((cmp["diff"] > 0.01).sum())
    hurt = int((cmp["diff"] < -0.01).sum())
    lines = [
        f"\n--- attribution: {name} vs baseline ---",
        cmp.to_string(float_format=lambda v: f"{v:+.3f}"),
        f"total outperformance: {total:+.3f}",
        f"top-2 years {list(top2.index)} = {top2.sum():+.3f}  "
        f"({'CONCENTRATED / overfit risk' if abs(top2.sum()) >= 0.8 * abs(total) and total != 0 else 'spread'})",
        f"years helped (>+1%): {helped}/{len(cmp)}   hurt (<-1%): {hurt}/{len(cmp)}",
    ]
    return "\n".join(lines)


def compare(
    start: str = "2010-03-01",
    end: str = "2024-12-31",
    initial_capital: float = 70_000.0,
) -> tuple[pd.DataFrame, dict[str, OverlayResult]]:
    cfg = V4Config()
    prices = fetch_etf_prices(start=start, end=end)
    spy_ohlc = _fetch_spy_ohlc(start, end)
    spy_adx = adx(spy_ohlc, 14).reindex(prices.index)

    results: dict[str, OverlayResult] = {}
    results["baseline"] = run_overlay_backtest(prices, cfg, "baseline", start=start, end=end, initial_capital=initial_capital)
    results["sma200"] = run_overlay_backtest(prices, cfg, "sma200", start=start, end=end, initial_capital=initial_capital)
    results["adx20"] = run_overlay_backtest(prices, cfg, "adx", spy_adx=spy_adx, adx_threshold=20.0, start=start, end=end, initial_capital=initial_capital)
    results["adx25"] = run_overlay_backtest(prices, cfg, "adx", spy_adx=spy_adx, adx_threshold=25.0, start=start, end=end, initial_capital=initial_capital)
    results["accel"] = run_accel_backtest(prices, cfg, start=start, end=end, initial_capital=initial_capital)

    table = pd.DataFrame([_row(name, res) for name, res in results.items()]).set_index("variant")
    return table, results


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    table, results = compare()
    print("\n=== V4 ETF short-term TA overlay comparison (2010-2024, selection params fixed) ===\n")
    print(table.to_string())

    base = table.loc["baseline"]
    base_eq = results["baseline"].equity
    print("\n--- recommendation ---")
    winners = []
    for name in table.index:
        if name == "baseline":
            continue
        r = table.loc[name]
        if (r["Sharpe"] > base["Sharpe"] or r["MaxDD"] > base["MaxDD"]) and r["GatesPass"]:
            winners.append(name)
            print(attribution(base_eq, results[name].equity, name))

    if not winners:
        print("\nNo overlay beats baseline on Sharpe/MaxDD while passing all gates.")
        print("Recommendation: KEEP monthly V4. No short-term TA overlay is justified.")
    else:
        print(f"\nTable-level winners: {winners}. Adopt ONLY if attribution above shows the "
              f"edge is 'spread', not 'CONCENTRATED'. Concentrated edges = overfit; reject.")


if __name__ == "__main__":
    main()
