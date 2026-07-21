"""Does *combining* technical-analysis conditions raise the win rate — or just
raise selectivity?

Three single-variable premises have already been probed and killed (static
support-hold, static breakdown-short, dynamic 20-EMA pullback), each with ~0
placebo-adjusted edge. The natural next question is whether *stacking* corroborating
TA conditions rescues the pullback entry. The trap: in a market that drifted up
2000–2019, every filter you add makes you more selective, which raises the raw
win rate **and its matched base rate together**. Raw win rate is therefore
uninterpretable; only the *lift over a locally-matched placebo* means anything.

The failed intraday TA engine (RSI + MACD + S&R, six attempts) is the cautionary
precedent — a combination that was never placebo-tested. This probe tests the
combination hypothesis directly, before any strategy code.

THE SETUP
---------
The candidate entry is a 20-EMA pullback touch in a strict uptrend (the exact,
already-sealed trigger from `trend_pullback_calibration`). Every touch is scored by
how many of five pre-registered, independent TA conditions align at t-1, and we ask
whether the placebo-adjusted 1R win rate rises with that confluence count.

THE PLACEBO — a locally-matched ghost
-------------------------------------
Each real touch is matched to a random bar in the *same ticker and same calendar
year* that resolves a valid 1R. Same-year matching captures the local drift and
volatility, so if high confluence merely selects favourable bull periods, the ghost
win rate rises alongside it and the lift stays flat. The signal is real only if the
*lift* (real − matched ghost), not the raw rate, rises with confluence.

LOOK-AHEAD SEALING
------------------
Every condition is read at t-1, never t. Regime run, RSI, MACD histogram and leg
stretch are causal series evaluated at t-1; support zones are built from a trailing
window strictly before t; entry is the open of t+1 and risk is sized from ATR(t-1).
`test_confluence_seal_does_not_leak` truncates the frame at t-1 and proves the
confluence score is bit-identical.

Window 2000-01-01 → 2019-12-31. 2020→now stays SEALED for a one-shot final
validation. SURVIVORSHIP CAVEAT: the universe is *current* S&P 500 membership applied
back to 2000, so the cohort is upward-biased and pre-2010 bars re-download from
yfinance (the cache starts 2010). Read the cross-bucket *gradient*, which is a
within-sample comparison and far less sensitive to that bias than any absolute rate.

Run: PYTHONPATH=. .venv/bin/python3 scripts/confluence_calibration.py
Env: CONFCAL_MAX_TICKERS, CONFCAL_START, CONFCAL_END
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from data.market import fetch_daily_ohlcv
from data.universe import apply_size_filter, get_sp500_tickers
from scripts.trend_pullback_calibration import (
    REGIME_BARS,
    _is_touch,
    _leg_max_stretch,
    _regime_run,
    _resolve_bounce,
    _two_proportion_z,
    trigger_lines,
)
from signals.indicators.momentum import macd, rsi
from signals.price_action.level_quality import build_zones

START = os.environ.get("CONFCAL_START", "2000-01-01")
END = os.environ.get("CONFCAL_END", "2019-12-31")   # 2020+ sealed for out-of-sample
MAX_TICKERS = int(os.environ.get("CONFCAL_MAX_TICKERS", "150"))

LOOKBACK = 500          # trailing bars used to build support zones
REBUILD_EVERY = 21      # zone rebuild cadence (~monthly)
MAX_HOLD = 40           # (mirrors the pullback resolver's horizon)

# Pre-registered confluence conditions (fixed set — no subset search, which would be
# multiple-comparisons overfitting). Each is a boolean read strictly at t-1.
TREND_MATURITY_BARS = 30   # C1: the 20>50>200 stack has held this long
RSI_MAX = 55.0             # C2: pullback has unwound momentum, not overbought
STRENGTH_MIN = 0.4         # C4: the touched level coincides with a real support zone
STRETCH_MAX = 3.0          # C5: a controlled pullback, not a climactic overextension
N_CONDITIONS = 5

GHOST_TRIES = 25        # attempts to find a resolvable same-year random bar
MIN_CELL_N = 200
RANDOM_SEED = 7


@dataclass
class ConfObs:
    ticker: str
    year: int
    confluence: int
    outcome: str          # "win" | "loss" | "neither"
    bars_to_1r: float | None
    is_placebo: bool


def _confluence_score(
    t: int, ema20: np.ndarray, atr_arr: np.ndarray, high: np.ndarray,
    run: np.ndarray, rsi_arr: np.ndarray, hist_arr: np.ndarray,
    zones: list, z_last: int, leg_start: int,
) -> int:
    """Count the pre-registered conditions that hold at t-1 (sealed)."""
    conds: list[bool] = []

    # C1 — trend maturity: the strict uptrend stack has held a while.
    conds.append(bool(run[t - 1] >= TREND_MATURITY_BARS))

    # C2 — momentum room: RSI has pulled back, not overbought.
    r = rsi_arr[t - 1]
    conds.append(bool(np.isfinite(r) and r <= RSI_MAX))

    # C3 — trend intact: MACD histogram non-negative (line at/above signal).
    h = hist_arr[t - 1]
    conds.append(bool(np.isfinite(h) and h >= 0.0))

    # C4 — S/R confluence: the touched level sits in a real support zone.
    line = ema20[t - 1]
    sr = False
    if zones and np.isfinite(line):
        for z in zones:
            if z.lo <= line <= z.hi and z.strength_as_of(z_last) >= STRENGTH_MIN:
                sr = True
                break
    conds.append(sr)

    # C5 — controlled pullback: the leg wasn't stretched to a climax.
    stretch = _leg_max_stretch(high, ema20, atr_arr, leg_start + 1, t)
    conds.append(bool(stretch <= STRETCH_MAX))

    return int(sum(conds))


def _matched_ghost(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, atr_arr: np.ndarray,
    years: np.ndarray, year: int, first: int, last: int, rng: random.Random,
) -> tuple[str, float | None] | None:
    """A random resolvable 1R in the same calendar year — the local base rate."""
    idx = np.flatnonzero((years == year) & (np.arange(len(years)) >= first)
                         & (np.arange(len(years)) < last))
    if len(idx) == 0:
        return None
    for _ in range(GHOST_TRIES):
        g = int(rng.choice(idx))
        res = _resolve_bounce(open_, high, low, atr_arr, g)
        if res is not None:
            return res
    return None


def collect(ticker: str, bars: pd.DataFrame, rng: random.Random) -> list[ConfObs]:
    obs: list[ConfObs] = []
    if len(bars) < LOOKBACK + MAX_HOLD + 2:
        return obs

    ema20, ema50, ema200, atr_arr = trigger_lines(bars)
    run = _regime_run(ema20, ema50, ema200)
    rsi_arr = rsi(bars["close"]).to_numpy(dtype=float)
    hist_arr = macd(bars["close"]).histogram.to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    years = bars.index.year.to_numpy()

    n = len(bars)
    first = LOOKBACK + 1          # need a full trailing window for zones
    last = n - MAX_HOLD - 2

    zones: list = []
    z_last = 0
    last_rebuild = -10_000

    leg_started = False
    leg_start = first

    for t in range(first, last):
        # Causal zone rebuild from a trailing window ending at t (never sees >= t).
        if t - last_rebuild >= REBUILD_EVERY:
            window = bars.iloc[t - LOOKBACK: t]
            try:
                zones = build_zones(window)
            except Exception as exc:
                logger.debug(f"{ticker}: zone build failed at {t}: {exc}")
                zones = []
            z_last = len(window) - 1
            last_rebuild = t

        if not _is_touch(low, high, ema20, t):
            continue

        if leg_started and run[t - 1] >= REGIME_BARS:
            resolved = _resolve_bounce(open_, high, low, atr_arr, t)
            if resolved is not None:
                c = _confluence_score(t, ema20, atr_arr, high, run, rsi_arr,
                                      hist_arr, zones, z_last, leg_start)
                obs.append(ConfObs(
                    ticker=ticker, year=int(years[t]), confluence=c,
                    outcome=resolved[0], bars_to_1r=resolved[1], is_placebo=False,
                ))
                ghost = _matched_ghost(open_, high, low, atr_arr, years,
                                       int(years[t]), first, last, rng)
                if ghost is not None:
                    obs.append(ConfObs(
                        ticker=ticker, year=int(years[t]), confluence=c,
                        outcome=ghost[0], bars_to_1r=ghost[1], is_placebo=True,
                    ))

        leg_started = True
        leg_start = t

    return obs


def _bucket(c: int) -> str:
    return "Low(0-1)" if c <= 1 else ("Mid(2-3)" if c <= 3 else "High(4-5)")


def report(df: pd.DataFrame) -> None:
    real = df[~df.is_placebo].copy()
    placebo = df[df.is_placebo].copy()
    if real.empty:
        print("No confluence observations collected.")
        return

    real["win"] = real.outcome == "win"
    placebo["win"] = placebo.outcome == "win"

    print("\n" + "=" * 92)
    print("  TA-CONFLUENCE CALIBRATION — does stacking conditions raise the placebo-adjusted win rate?")
    print("=" * 92)
    print(f"  Window {START} → {END}   tickers={real.ticker.nunique()}   "
          f"touches={len(real)}  ghosts={len(placebo)}")
    print(f"  Entry = 20-EMA pullback in a strict uptrend. Confluence = # of {N_CONDITIONS} "
          f"conditions met at t-1.")
    print("  Ghost = random same-year bar in the same name (local drift/vol control).\n")

    # Per exact confluence level.
    rows = []
    for c in range(N_CONDITIONS + 1):
        g = real[real.confluence == c]
        pg = placebo[placebo.confluence == c]
        if len(g) == 0:
            continue
        wr = g.win.mean()
        pr = pg.win.mean() if len(pg) else float("nan")
        rows.append({
            "confluence": c, "n": len(g),
            "win_rate": wr,
            "placebo_win_rate": pr,
            "lift": wr - pr if len(pg) else float("nan"),
            "median_bars_1R": g.loc[g.win, "bars_to_1r"].median(),
            "reliable": "" if len(g) >= MIN_CELL_N else "  LOW-N",
        })
    table = pd.DataFrame(rows).set_index("confluence")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    # Coarse buckets for the gates (ensure cell size).
    real["bucket"] = real.confluence.map(_bucket)
    placebo["bucket"] = placebo.confluence.map(_bucket)
    order = ["Low(0-1)", "Mid(2-3)", "High(4-5)"]

    def _rate(frame, b):
        s = frame[frame.bucket == b]
        return (s.win.mean(), int(s.win.sum()), len(s))

    print("\n  Coarse buckets:")
    brows = []
    for b in order:
        rwr, rw, rn = _rate(real, b)
        pwr, _, pn = _rate(placebo, b)
        brows.append({"bucket": b, "n": rn, "win_rate": rwr,
                      "placebo_win_rate": pwr,
                      "lift": (rwr - pwr) if pn else float("nan")})
    btable = pd.DataFrame(brows).set_index("bucket")
    print(btable.to_string(float_format=lambda v: f"{v:,.3f}"))

    # ── Kill criteria ──
    present = [b for b in order if _rate(real, b)[2] > 0]
    top, bot = present[-1], present[0]
    top_wr, top_w, top_n = _rate(real, top)
    bot_wr, bot_w, bot_n = _rate(real, bot)
    top_pwr, top_pw, top_pn = _rate(placebo, top)
    top_lift = top_wr - top_pwr if top_pn else float("nan")
    top_lift_p = _two_proportion_z(top_w, top_n, top_pw, top_pn)

    raw_grad = top_wr - bot_wr
    lift_grad = btable.loc[top, "lift"] - btable.loc[bot, "lift"]
    rho = btable["lift"].corr(pd.Series(range(len(btable)), index=btable.index),
                              method="spearman")

    print("\n" + "-" * 92)
    print("  KILL CRITERIA")
    print("-" * 92)
    c1 = (not math.isnan(top_lift)) and top_lift >= 0.05 and top_lift_p < 0.05
    c2 = (not math.isnan(rho)) and rho >= 0.6 and lift_grad >= 0.05
    mirage = raw_grad >= 0.05 and (math.isnan(lift_grad) or lift_grad <= 0.02)
    c3 = not mirage
    print(f"  1. Top-bucket lift >= 5pp, p<0.05                    : "
          f"{top_lift*100:+.1f}pp  p={top_lift_p:.4f}   {'PASS' if c1 else 'FAIL'}")
    print(f"  2. Lift rises with confluence (rho>=0.6, grad>=5pp)  : "
          f"rho={rho:+.2f}  grad={lift_grad*100:+.1f}pp   {'PASS' if c2 else 'FAIL'}")
    print(f"  3. Not a selectivity mirage (raw up but lift flat)   : "
          f"raw_grad={raw_grad*100:+.1f}pp lift_grad={lift_grad*100:+.1f}pp   "
          f"{'PASS' if c3 else 'FAIL'}")

    if mirage:
        print("\n  !! MIRAGE — raw win rate rises with confluence but the placebo-adjusted lift")
        print("     does not. The 'edge' is pure selectivity in a drifting market, not signal.")
    if not math.isnan(top_lift) and top_lift > 0.15:
        print("\n  !! Lift exceeds 15pp — treat as a look-ahead bug until proven otherwise.")

    print("\n  " + ("PROCEED — combining TA raises the placebo-adjusted win rate. Design the engine."
                    if (c1 and c2 and c3)
                    else "STOP. Record the outcome in DECISIONS.md."))


def main() -> None:
    if END > "2019-12-31":
        raise SystemExit(
            f"CONFCAL_END={END} breaks the seal — 2020+ is strictly out-of-sample.")

    rng = random.Random(RANDOM_SEED)
    tickers = get_sp500_tickers()[:MAX_TICKERS]
    logger.info(f"Confluence calibration: {len(tickers)} tickers, {START} → {END}")

    bars = fetch_daily_ohlcv(tickers, START, END)
    bars = apply_size_filter(bars)

    all_obs: list[ConfObs] = []
    for n, (ticker, ohlcv) in enumerate(sorted(bars.items()), 1):
        try:
            all_obs.extend(collect(ticker, ohlcv, rng))
        except Exception as exc:                       # one bad ticker must not kill the study
            logger.debug(f"{ticker}: collect failed: {exc}")
        if n % 25 == 0:
            logger.info(f"  {n}/{len(bars)} tickers, {len(all_obs)} observations")

    if not all_obs:
        print("No observations collected — check the universe and date range.")
        return
    report(pd.DataFrame([o.__dict__ for o in all_obs]))


if __name__ == "__main__":
    main()
