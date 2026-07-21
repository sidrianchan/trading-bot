"""The "Beach Ball" thesis: when SPY pulls back and reverses, do top relative-
strength names snap back harder than market-beta and laggards?

Four single-stock price-action-geometry premises have now been probed and killed
(static support-hold, static breakdown-short, dynamic 20-EMA pullback, TA
confluence). This is the pivot to an *orthogonal, cross-sectional* signal: a
ticker's **relative strength** — its rank against its peers on a given day — rather
than any pattern in its own price history.

The setup. SPY makes a local pullback (a drawdown off its 10-bar high) and then
prints a reversal bar. On that reversal we ask: of the stocks ranked by trailing
10-bar excess return over SPY, do the top-decile (D10) names win a symmetric 1R
more often than the median (D5, ~market beta) and the bottom (D1)? The cross-section
is its own control — D5 is the beta baseline and D1 the laggard baseline, so no
ghost placebo is needed here.

Still strictly a measurement probe: no backtester, no strategy engine, no scoring.

LOOK-AHEAD SEALING
------------------
1. **RS ranks use only t-1**: excess return is ln(P[t-1]/P[t-11]) for the stock
   minus the same for SPY — every price at or before t-1. Ranked cross-sectionally
   over the t-1 cross-section only.
2. **The SPY pullback filter is evaluated at t-1**; the reversal *fires* on day t
   (that is the trigger, and the only thing allowed to read day-t data). Day t never
   enters the ranking.
3. **ATR is read at t-1**, entry is the open of t+1, risk is symmetric ±1 ATR(t-1).

STATISTICAL CAVEAT
------------------
Events are heavily cross-sectionally clustered: one SPY reversal fires hundreds of
simultaneous events, and reversals themselves cluster in time. The two-proportion
z-test below treats those correlated events as independent, so its p-values are
**anti-conservative** — a passing p<0.01 is weaker evidence than the number implies.
Read the gradient's *size* and *monotonicity*, not the p-value alone.

Window 2000-01-01 → 2019-12-31. 2020→now stays SEALED. SURVIVORSHIP CAVEAT: the
universe is *current* S&P 500 membership applied back to 2000, biasing the cohort
upward and re-downloading pre-2010 bars from yfinance.

Run: PYTHONPATH=. .venv/bin/python3 scripts/relative_strength_calibration.py
Env: RSCAL_MAX_TICKERS, RSCAL_START, RSCAL_END
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from data.market import fetch_daily_ohlcv
from data.universe import apply_size_filter, get_sp500_tickers
from signals.indicators.volatility import atr

START = os.environ.get("RSCAL_START", "2000-01-01")
END = os.environ.get("RSCAL_END", "2019-12-31")   # 2020+ sealed for out-of-sample
MAX_TICKERS = int(os.environ.get("RSCAL_MAX_TICKERS", "150"))

ATR_PERIOD = 20
RS_LOOKBACK = 10          # bars in the excess-return window
PULLBACK_WINDOW = 10      # SPY drawdown window [t-10, t-1]
PULLBACK_ATR = 1.5        # SPY must have dropped this many ATRs off the window high
MAX_HOLD = 40             # bars to resolve the 1R
N_DECILES = 10
MIN_ACTIVE = N_DECILES    # need at least this many active names to rank into deciles
MIN_CELL_N = 200


@dataclass
class RSObs:
    ticker: str
    year: int
    decile: int
    outcome: str          # "win" | "loss" | "neither"
    bars_to_1r: float | None


def _spy_pullback_ok(high: np.ndarray, close: np.ndarray, atr_arr: np.ndarray, t: int) -> bool:
    """SPY dropped >= PULLBACK_ATR*ATR20 off its 10-bar high, as of t-1."""
    if t < PULLBACK_WINDOW + 1:
        return False
    a = atr_arr[t - 1]
    if not np.isfinite(a) or a <= 0:
        return False
    peak = np.nanmax(high[t - PULLBACK_WINDOW: t])   # bars t-10 .. t-1
    c = close[t - 1]
    if not (np.isfinite(peak) and np.isfinite(c)):
        return False
    return (peak - c) >= PULLBACK_ATR * a


def _spy_reversal(open_: np.ndarray, close: np.ndarray, high: np.ndarray, t: int) -> bool:
    """Day-t reversal: closes green OR closes above the prior high."""
    if t < 1:
        return False
    c, o, ph = close[t], open_[t], high[t - 1]
    if not (np.isfinite(c) and np.isfinite(o) and np.isfinite(ph)):
        return False
    return bool(c > o or c > ph)


def _excess_returns(close_mat: np.ndarray, spy_close: np.ndarray, t: int) -> np.ndarray:
    """Per-ticker trailing-10-bar log excess return over SPY, evaluated at t-1.

    Returns a vector over all tickers with NaN where the stock is inactive (missing
    price at t-1 or t-1-RS_LOOKBACK). SPY's own window must be finite and positive.
    """
    i1 = t - 1
    i0 = t - 1 - RS_LOOKBACK
    if i0 < 0:
        return np.full(close_mat.shape[1], np.nan)
    sp1, sp0 = spy_close[i1], spy_close[i0]
    if not (np.isfinite(sp1) and np.isfinite(sp0) and sp1 > 0 and sp0 > 0):
        return np.full(close_mat.shape[1], np.nan)
    p1 = close_mat[i1]
    p0 = close_mat[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        stock = np.log(p1 / p0)
        excess = stock - math.log(sp1 / sp0)
    excess[~np.isfinite(p1) | ~np.isfinite(p0) | (p1 <= 0) | (p0 <= 0)] = np.nan
    return excess


def _assign_deciles(excess: np.ndarray) -> np.ndarray:
    """Cross-sectional deciles 1..10 (D1 lowest, D10 highest); NaN where inactive."""
    out = np.full(len(excess), np.nan)
    mask = np.isfinite(excess)
    if mask.sum() < MIN_ACTIVE:
        return out
    ser = pd.Series(excess[mask])
    labels = pd.qcut(ser.rank(method="first"), N_DECILES, labels=range(1, N_DECILES + 1))
    out[np.flatnonzero(mask)] = labels.to_numpy(dtype=float)
    return out


def _resolve_1r(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, atr_arr: np.ndarray, t: int,
) -> tuple[str, float | None] | None:
    """Symmetric 1R long from a trigger at t; gap-aware forward resolution.

    entry = open[t+1]; target = entry + ATR(t-1); stop = entry - ATR(t-1).
    Per bar j: gap first (open<=stop -> loss; open>=target -> win) before the
    high/low check; a same-bar span of both target and stop is a conservative loss.
    """
    n = len(open_)
    if t + 1 >= n or t < 1:
        return None
    a = atr_arr[t - 1]
    entry = open_[t + 1]
    if not (np.isfinite(a) and a > 0 and np.isfinite(entry)):
        return None
    target = entry + a
    stop = entry - a

    end = min(t + 1 + MAX_HOLD, n)
    for j in range(t + 1, end):
        oj, hj, lj = open_[j], high[j], low[j]
        if not (np.isfinite(oj) and np.isfinite(hj) and np.isfinite(lj)):
            return "neither", None            # delisted mid-trade
        # Gap check precedes the intrabar high/low check.
        if oj <= stop:
            return "loss", None
        if oj >= target:
            return "win", float(j - t)
        hit_target = hj >= target
        hit_stop = lj <= stop
        if hit_target and hit_stop:
            return "loss", None               # collision -> conservative loss
        if hit_target:
            return "win", float(j - t)
        if hit_stop:
            return "loss", None
    return "neither", None


def _two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided p-value for a difference in proportions (no scipy)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = s1 / n1, s2 / n2
    pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return float(math.erfc(z / math.sqrt(2.0)))


def _build_panel(
    bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align every ticker onto the SPY calendar as [n_dates, n_tickers] matrices."""
    tickers = sorted(bars)
    cols_o, cols_h, cols_l, cols_c, cols_a = [], [], [], [], []
    for tk in tickers:
        df = bars[tk].sort_index()
        a = atr(df, period=ATR_PERIOD)
        r = df.reindex(index)
        cols_o.append(r["open"].to_numpy(dtype=float))
        cols_h.append(r["high"].to_numpy(dtype=float))
        cols_l.append(r["low"].to_numpy(dtype=float))
        cols_c.append(r["close"].to_numpy(dtype=float))
        cols_a.append(a.reindex(index).to_numpy(dtype=float))
    def stack(cols: list) -> np.ndarray:
        return np.column_stack(cols) if cols else np.empty((len(index), 0))
    return (tickers, stack(cols_o), stack(cols_h), stack(cols_l), stack(cols_c), stack(cols_a))


def collect(
    tickers: list[str],
    open_mat: np.ndarray, high_mat: np.ndarray, low_mat: np.ndarray,
    close_mat: np.ndarray, atr_mat: np.ndarray,
    spy_open: np.ndarray, spy_high: np.ndarray, spy_low: np.ndarray,
    spy_close: np.ndarray, spy_atr: np.ndarray, years: np.ndarray,
) -> list[RSObs]:
    obs: list[RSObs] = []
    n = len(spy_close)
    for t in range(PULLBACK_WINDOW + 1, n - MAX_HOLD - 1):
        if not _spy_pullback_ok(spy_high, spy_close, spy_atr, t):
            continue
        if not _spy_reversal(spy_open, spy_close, spy_high, t):
            continue

        deciles = _assign_deciles(_excess_returns(close_mat, spy_close, t))
        active = np.flatnonzero(np.isfinite(deciles))
        if len(active) == 0:
            continue

        year = int(years[t])
        for i in active:
            resolved = _resolve_1r(open_mat[:, i], high_mat[:, i], low_mat[:, i],
                                   atr_mat[:, i], t)
            if resolved is None:
                continue
            obs.append(RSObs(
                ticker=tickers[i], year=year, decile=int(deciles[i]),
                outcome=resolved[0], bars_to_1r=resolved[1],
            ))
    return obs


def report(df: pd.DataFrame) -> None:
    df = df.copy()
    df["win"] = df.outcome == "win"

    print("\n" + "=" * 88)
    print("  RELATIVE-STRENGTH CALIBRATION — do top-RS names snap back harder on a SPY reversal?")
    print("=" * 88)
    print(f"  Window {START} → {END}   tickers={df.ticker.nunique()}   events={len(df)}")
    print("  Signal = trailing-10-bar excess return vs SPY, ranked cross-sectionally at t-1.")
    print("  Entry = open after a SPY pullback-and-reversal; symmetric 1R = ±1 ATR20(t-1).\n")

    rate = {}
    med = {}
    wins = {}
    ns = {}
    rows = []
    for d in range(1, N_DECILES + 1):
        g = df[df.decile == d]
        rate[d] = g.win.mean() if len(g) else float("nan")
        med[d] = g.loc[g.win, "bars_to_1r"].median() if len(g) else float("nan")
        wins[d] = int(g.win.sum())
        ns[d] = len(g)
        rows.append({
            "decile": d, "n_events": len(g), "win_rate": rate[d],
            "lift_vs_D5": float("nan"), "median_bars_to_1R": med[d],
            "reliable": "" if len(g) >= MIN_CELL_N else "  LOW-N",
        })
    table = pd.DataFrame(rows).set_index("decile")
    d5 = rate.get(5, float("nan"))
    table["lift_vs_D5"] = table["win_rate"] - d5
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    # ── Kill criteria ──
    grad = rate[10] - rate[1]
    grad_p = _two_proportion_z(wins[10], ns[10], wins[1], ns[1])
    lift_beta = rate[10] - rate[5]
    speed_ok = (not math.isnan(med[10])) and (not math.isnan(med[5])) and med[10] < med[5]
    rho = table["win_rate"].corr(pd.Series(table.index, index=table.index), method="spearman")

    print("\n" + "-" * 88)
    print("  KILL CRITERIA")
    print("-" * 88)
    c1 = grad >= 0.08 and grad_p < 0.01
    c2 = lift_beta >= 0.05
    c3 = speed_ok
    c4 = (not math.isnan(rho)) and rho >= 0.70
    print(f"  1. Cross-sectional gradient D10-D1 >= 8pp, p<0.01   : "
          f"{grad*100:+.1f}pp  p={grad_p:.4f}   {'PASS' if c1 else 'FAIL'}")
    print(f"  2. Lift over beta D10-D5 >= 5pp                     : "
          f"{lift_beta*100:+.1f}pp               {'PASS' if c2 else 'FAIL'}")
    print(f"  3. Speed to target median_bars(D10) < median(D5)   : "
          f"{med[10]:.1f} vs {med[5]:.1f} bars        {'PASS' if c3 else 'FAIL'}")
    print(f"  4. Monotonicity Spearman rho(decile, win) >= 0.70  : "
          f"rho={rho:+.3f}             {'PASS' if c4 else 'FAIL'}")

    print("\n  NOTE: events are cross-sectionally clustered (one SPY reversal fires hundreds of")
    print("  simultaneous events; reversals cluster in time), so the z-test p-value is")
    print("  anti-conservative. Weight the gradient's SIZE and MONOTONICITY over the p-value.")
    if not math.isnan(lift_beta) and lift_beta > 0.15:
        print("\n  !! D10 lift over beta exceeds 15pp — treat as a look-ahead bug until proven otherwise.")

    print("\n  " + ("PROCEED — cross-sectional RS carries a reversal edge. Design the setup engine."
                    if (c1 and c2 and c3 and c4)
                    else "STOP. Record the outcome in DECISIONS.md."))


def main() -> None:
    if END > "2019-12-31":
        raise SystemExit(
            f"RSCAL_END={END} breaks the seal — 2020+ is strictly out-of-sample.")

    logger.info(f"Relative-strength calibration: up to {MAX_TICKERS} tickers, {START} → {END}")
    spy_map = fetch_daily_ohlcv(["SPY"], START, END)
    if "SPY" not in spy_map or spy_map["SPY"].empty:
        print("SPY data unavailable — cannot compute excess returns.")
        return
    spy = spy_map["SPY"].sort_index()
    index = spy.index

    tickers = get_sp500_tickers()[:MAX_TICKERS]
    bars = fetch_daily_ohlcv(tickers, START, END)
    bars.pop("SPY", None)                       # SPY is the benchmark, not a candidate
    bars = apply_size_filter(bars)
    if not bars:
        print("No universe tickers survived the size filter.")
        return

    names, o, h, low_, c, a = _build_panel(bars, index)
    spy_atr = atr(spy, period=ATR_PERIOD).to_numpy(dtype=float)
    years = index.year.to_numpy()

    logger.info(f"Panel: {len(names)} tickers × {len(index)} bars; scanning SPY triggers")
    obs = collect(
        names, o, h, low_, c, a,
        spy["open"].to_numpy(dtype=float), spy["high"].to_numpy(dtype=float),
        spy["low"].to_numpy(dtype=float), spy["close"].to_numpy(dtype=float),
        spy_atr, years,
    )
    if not obs:
        print("No events collected — check the universe and date range.")
        return
    report(pd.DataFrame([o.__dict__ for o in obs]))


if __name__ == "__main__":
    main()
