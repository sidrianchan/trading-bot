"""Does level strength predict whether a *breakdown* follows through?

A companion probe to `scripts/level_calibration.py`, which established that
support-zone strength predicts *holding* only weakly (a real +3.4pp tercile
spread, but too small to trade). That study surfaced one sharp anomaly worth its
own measurement: the very strongest zones — decile 10, the most obvious,
heavily-touched, high-volume support — *held less often* than moderate zones
(61.7% vs ~65%). The hypothesis this script tests is the short-side mirror of
that finding: **when an obvious, strong support finally breaks, does price flush,
because that is exactly where the resting stops and liquidity sit?**

This is still a measurement study. No backtester, no setup engine, no scoring.
It moves the observation window from the *touch* (level_calibration) to the
*break*, and asks one question per strength decile: of the zones that broke down
cleanly, what fraction flushed a full 1R before the break failed?

THE PLACEBO — controlling for downside drift
--------------------------------------------
2010–2019 drifted strongly upward, which *deflates* any short-side flush rate
uniformly. So an absolute flush rate is uninterpretable, exactly as the hold rate
was in the long study. For every real break we place a structureless "ghost band"
of identical width at a random 0.5–5 ATR below the price, wait for price to break
*it* under identical volume/volatility rules, and measure the same outcome. The
ghost has no support behind it, so its flush rate is the base rate of "short any
random breakdown in this name, at this time." The signal is the *difference*.

LOOK-AHEAD SEALING
------------------
1. **Zone strength is sealed at the last rebuild**, strictly before the break.
   Zones are built from a trailing window ending at bar t-1; the break is found
   on bars >= t that the builder never saw. `strength_as_of(last)` therefore uses
   only in-window information — sealed by construction, and conservatively so.
2. **ATR and average volume are read at bar b-1**, never bar b. The break bar is a
   high-range, high-volume bar by definition; sizing risk from its *own* range or
   averaging in its *own* volume would leak the trigger into the measurement.
3. **Entry is the open of bar b+1**, never the close of the break bar — the first
   price a short could actually be filled at after the signal prints.

Run: .venv/bin/python3 scripts/breakdown_calibration.py
Env: BREAKCAL_MAX_TICKERS, BREAKCAL_START, BREAKCAL_END
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
from signals.indicators.trend import classify
from signals.indicators.volatility import atr
from signals.price_action.level_quality import build_zones

START = os.environ.get("BREAKCAL_START", "2010-01-01")
END = os.environ.get("BREAKCAL_END", "2019-12-31")   # 2020+ sealed for out-of-sample
MAX_TICKERS = int(os.environ.get("BREAKCAL_MAX_TICKERS", "150"))

LOOKBACK = 500          # trailing bars used to construct zones
REBUILD_EVERY = 21      # rebuild cadence (~monthly)
BREAK_HORIZON = 21      # bars after a rebuild in which a fresh break may occur
MAX_HOLD = 40           # bars over which the breakdown outcome is resolved
SWEEP_WINDOW = 5        # a bear trap must reclaim within this many bars of the break
GHOST_HORIZON = 60      # bars a ghost band gets to be broken before we give up

BREAK_CLOSE_ATR = 0.5   # close must clear the band low by this to count as broken
BREAK_VOLUME_RATIO = 1.3   # ...on at least this much of the trailing 20-day average
STOP_ATR = 0.2          # a close back above hi + this*ATR fails the breakdown
GHOST_MIN_OFFSET = 0.5  # ghost band placed 0.5..5 ATR below the price
GHOST_MAX_OFFSET = 5.0

MIN_CELL_N = 200
RANDOM_SEED = 7


@dataclass
class BreakObs:
    ticker: str
    year: int
    strength: float
    regime: str
    outcome: str          # "flush" | "sweep" | "neither"
    bars_to_1r: float | None
    is_placebo: bool


def _first_break(
    close: np.ndarray, volume: np.ndarray, atr_arr: np.ndarray, vol_avg: np.ndarray,
    start: int, horizon: int, lo: float, hi: float,
) -> int | None:
    """First bar in [start, start+horizon) that closes cleanly below a band.

    A break requires price to have been *engaged* with the band (closed at or
    above its low at some prior bar) so a level already far below price cannot
    register a spurious break, and requires a real-volume close beyond it. ATR
    and the volume average are read at b-1 so the break bar never sizes its own
    threshold — the same seal used everywhere in this study.
    """
    n = len(close)
    end = min(start + horizon, n)
    engaged = False
    for b in range(start, end):
        if b < 1:
            continue
        a = atr_arr[b - 1]
        avg_v = vol_avg[b - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        if close[b] >= lo:
            engaged = True
        if not engaged:
            continue
        if close[b] < lo - BREAK_CLOSE_ATR * a:
            vr = float(volume[b] / avg_v) if avg_v and np.isfinite(avg_v) else 1.0
            if vr >= BREAK_VOLUME_RATIO:
                return b
            engaged = False   # a low-volume poke; wait for re-engagement
    return None


def _resolve_break(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    atr_arr: np.ndarray, b: int, lo: float, hi: float,
) -> tuple[str, float | None] | None:
    """Forward-resolve one clean breakdown into flush / sweep / neither.

    Short mechanics, all sealed relative to the break bar b:
      entry  = open[b+1]                       (first fillable price after signal)
      stop   = hi + STOP_ATR * ATR(b-1)        (a close back above fails the break)
      risk   = stop - entry                    ( > 0: entry is below the zone )
      target = entry - risk                    (1R down)

    A *flush* reaches the 1R target (intraday low) before any close back above the
    stop. A *sweep* is a bear trap: a close back above the stop within SWEEP_WINDOW
    bars, before the target is reached. Everything else is neither.
    """
    n = len(close)
    if b + 1 >= n or b < 1:
        return None
    a = atr_arr[b - 1]
    if not np.isfinite(a) or a <= 0:
        return None

    entry = float(open_[b + 1])
    stop = hi + STOP_ATR * a
    risk = stop - entry
    if risk <= 0:                 # entry gapped above the stop; not a short
        return None
    target = entry - risk

    end = min(b + 1 + MAX_HOLD, n)
    flush_bar: int | None = None
    stop_bar: int | None = None
    for j in range(b + 1, end):
        if flush_bar is None and low[j] <= target:
            flush_bar = j
        if stop_bar is None and close[j] >= stop:
            stop_bar = j
        if flush_bar is not None or stop_bar is not None:
            break

    if flush_bar is not None and (stop_bar is None or flush_bar <= stop_bar):
        return "flush", float(flush_bar - b)
    if stop_bar is not None and (stop_bar - b) <= SWEEP_WINDOW and (
        flush_bar is None or stop_bar < flush_bar
    ):
        return "sweep", None
    return "neither", None


def collect(ticker: str, bars: pd.DataFrame, rng: random.Random) -> list[BreakObs]:
    obs: list[BreakObs] = []
    if len(bars) < LOOKBACK + BREAK_HORIZON + MAX_HOLD:
        return obs

    atr_arr = atr(bars, period=20).to_numpy(dtype=float)
    vol_avg = bars["volume"].rolling(20, min_periods=5).mean().to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    years = bars.index.year.to_numpy()

    for t in range(LOOKBACK, len(bars) - MAX_HOLD - BREAK_HORIZON, REBUILD_EVERY):
        window = bars.iloc[t - LOOKBACK : t]         # builder sees nothing >= t
        try:
            zones = build_zones(window)
        except Exception as exc:
            logger.debug(f"{ticker}: zone build failed at {t}: {exc}")
            continue
        if not zones:
            continue

        last = len(window) - 1
        regime_snap = classify(window)
        regime = regime_snap.bias if regime_snap else "unknown"

        for zone in zones:
            strength = zone.strength_as_of(last)     # sealed at the rebuild
            if strength <= 0.0:
                continue

            b = _first_break(close, volume, atr_arr, vol_avg, t, BREAK_HORIZON, zone.lo, zone.hi)
            if b is None:
                continue
            resolved = _resolve_break(open_, high, low, close, atr_arr, b, zone.lo, zone.hi)
            if resolved is None:
                continue
            outcome, b1 = resolved
            obs.append(BreakObs(
                ticker=ticker, year=int(years[b]), strength=strength, regime=regime,
                outcome=outcome, bars_to_1r=b1, is_placebo=False,
            ))

            # Matched ghost band: identical width, random 0.5-5 ATR below the
            # break price, no structure behind it. Broken under identical rules.
            g = _ghost_break(close, volume, atr_arr, vol_avg, b, zone.hi - zone.lo, rng)
            if g is not None:
                g_b, g_lo, g_hi = g
                g_res = _resolve_break(open_, high, low, close, atr_arr, g_b, g_lo, g_hi)
                if g_res is not None:
                    obs.append(BreakObs(
                        ticker=ticker, year=int(years[g_b]), strength=strength,
                        regime=regime, outcome=g_res[0], bars_to_1r=g_res[1],
                        is_placebo=True,
                    ))
    return obs


def _ghost_break(
    close: np.ndarray, volume: np.ndarray, atr_arr: np.ndarray, vol_avg: np.ndarray,
    b: int, width: float, rng: random.Random,
) -> tuple[int, float, float] | None:
    """Place a structureless band below the break price and find where it breaks.

    The anchor is a random 0.5-5 ATR below the break bar's close, sized by ATR at
    b-1 (sealed). Returns (break_bar, lo, hi) of the ghost, or None if price never
    breaks it inside GHOST_HORIZON.
    """
    a = atr_arr[b - 1] if b >= 1 else np.nan
    if not np.isfinite(a) or a <= 0 or width <= 0:
        return None
    offset = rng.uniform(GHOST_MIN_OFFSET, GHOST_MAX_OFFSET)
    anchor = float(close[b]) - offset * a
    lo, hi = anchor - width / 2.0, anchor + width / 2.0
    if lo <= 0:
        return None
    gb = _first_break(close, volume, atr_arr, vol_avg, b + 1, GHOST_HORIZON, lo, hi)
    if gb is None:
        return None
    return gb, lo, hi


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


def report(df: pd.DataFrame) -> None:
    real = df[~df.is_placebo].copy()
    placebo = df[df.is_placebo].copy()
    if real.empty:
        print("No breaks collected.")
        return

    real["decile"] = pd.qcut(real["strength"].rank(method="first"), 10, labels=False) + 1
    real["flush"] = real.outcome == "flush"
    real["sweep"] = real.outcome == "sweep"
    if not placebo.empty:
        # Placebos inherit the decile of the real break they were matched to.
        placebo = placebo.merge(
            real.reset_index()[["ticker", "strength", "decile"]].drop_duplicates(
                subset=["ticker", "strength"]),
            on=["ticker", "strength"], how="left",
        )
        placebo["flush"] = placebo.outcome == "flush"

    print("\n" + "=" * 82)
    print("  BREAKDOWN CALIBRATION — does level strength predict a follow-through flush?")
    print("=" * 82)
    print(f"  Window {START} → {END}   tickers={real.ticker.nunique()}   "
          f"breaks={len(real)}  ghosts={len(placebo)}")
    print(f"  Zones built causally from a trailing {LOOKBACK}-bar window; breaks found on "
          f"unseen bars.\n")

    p_flush_global = placebo.flush.mean() if not placebo.empty else float("nan")

    rows = []
    for d, g in real.groupby("decile"):
        pg = placebo[placebo.decile == d] if not placebo.empty else placebo
        rows.append({
            "decile": int(d), "n": len(g),
            "strength_lo": g.strength.min(), "strength_hi": g.strength.max(),
            "flush_rate": g.flush.mean(),
            "sweep_rate": g.sweep.mean(),
            "placebo_flush": pg.flush.mean() if len(pg) else float("nan"),
            "med_bars_1R": g.loc[g.flush, "bars_to_1r"].median(),
            "reliable": "" if len(g) >= MIN_CELL_N else "  LOW-N",
        })
    table = pd.DataFrame(rows).set_index("decile")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    print(f"\n  Ghost band (random level, no structure): flush rate {p_flush_global:.3f}  "
          f"n={len(placebo)}")
    print("  Short-side drift deflates every rate uniformly — read the SPREAD and the LIFT.")

    top = real[real.decile >= 8]
    bottom = real[real.decile <= 3]
    spread = top.flush.mean() - bottom.flush.mean()
    pval = _two_proportion_z(int(top.flush.sum()), len(top), int(bottom.flush.sum()), len(bottom))

    d10 = real[real.decile == 10]
    d10_flush = d10.flush.mean()
    d10_sweep = d10.sweep.mean()
    lift = d10_flush - p_flush_global if not placebo.empty else float("nan")
    rho = table["flush_rate"].corr(pd.Series(table.index, index=table.index), method="spearman")

    print("\n" + "-" * 82)
    print("  KILL CRITERIA")
    print("-" * 82)
    c1 = spread >= 0.05 and pval < 0.05
    c2 = lift >= 0.05
    c3 = d10_sweep <= d10_flush
    print(f"  1. Tercile spread flush(D8-10) - flush(D1-3) >= 5pp, p<0.05 : "
          f"{spread*100:+.1f}pp  p={pval:.4f}   {'PASS' if c1 else 'FAIL'}")
    print(f"  2. D10 flush beats ghost-band flush by >= 5pp              : "
          f"{lift*100:+.1f}pp               {'PASS' if c2 else 'FAIL'}")
    print(f"  3. D10 flush >= D10 sweep (momentum-short, not bear-trap)  : "
          f"flush {d10_flush*100:.1f}% vs sweep {d10_sweep*100:.1f}%   "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"     (informational) Spearman rho(decile, flush rate)        : {rho:+.3f}")

    if not c3:
        print("\n  !! D10 sweep rate exceeds its flush rate: breaks of strong support are bear")
        print("     traps more often than momentum shorts. This KILLS the breakdown-short")
        print("     premise but VALIDATES a 'buy the reclaim' (long the sweep) premise.")
    if lift > 0.15:
        print("\n  !! Lift exceeds 15pp — treat as a look-ahead bug until proven otherwise.")

    print("\n  " + ("PROCEED — the breakdown-short edge is real. Design the setup engine."
                    if (c1 and c2 and c3)
                    else "STOP. Record the outcome in DECISIONS.md."))

    print("\n  By regime:")
    print(real.groupby("regime").agg(n=("flush", "size"), flush=("flush", "mean"),
                                     sweep=("sweep", "mean"))
          .to_string(float_format=lambda v: f"{v:,.3f}"))


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    tickers = get_sp500_tickers()[:MAX_TICKERS]
    logger.info(f"Breakdown calibration: {len(tickers)} tickers, {START} → {END}")

    bars = fetch_daily_ohlcv(tickers, START, END)
    bars = apply_size_filter(bars)

    all_obs: list[BreakObs] = []
    for n, (ticker, ohlcv) in enumerate(sorted(bars.items()), 1):
        all_obs.extend(collect(ticker, ohlcv, rng))
        if n % 25 == 0:
            logger.info(f"  {n}/{len(bars)} tickers, {len(all_obs)} observations")

    if not all_obs:
        print("No observations collected — check the universe and date range.")
        return
    report(pd.DataFrame([o.__dict__ for o in all_obs]))


if __name__ == "__main__":
    main()
