"""Does level strength predict whether support holds?

This is a measurement study, not a strategy. There is no backtester, no scoring
engine, no position sizing. It answers one question with one number, because six
prior attempts at support/resistance trading failed and DECISIONS.md records the
root cause as an inability to "distinguish 'support holding' from 'support
breaking'". If strength carries no information, we stop here and finally know
why — which is worth more than a seventh failed iteration.

TWO DEFENCES AGAINST LOOK-AHEAD
-------------------------------
1. **Causal zone construction.** Zones are rebuilt every REBUILD_EVERY bars from
   a trailing window of *past* bars only, then outcomes are measured on bars the
   builder never saw. Building once over the full sample would have been
   cheaper, but zone existence requires >=2 touches, so a level that broke on
   first touch and was never revisited would never become a zone — quietly
   selecting the sample toward levels that held.
2. **The sealing rule** inside `ZoneTrack.strength_as_of` (defence in depth).

CONTROL FOR SURVIVORSHIP BIAS
-----------------------------
The universe is current index membership, so it is survivorship-biased upward
and absolute hold rates are inflated in every decile — an absolute number here
is uninterpretable. Every real touch is therefore paired with a placebo: the
same ticker, a random date in the same calendar year, an identical-width band at
that bar's low with no structural claim behind it. Drift, regime and volatility
hit treatment and control equally, so the *difference* is drift-neutral.

Run: .venv/bin/python3 scripts/level_calibration.py
Env: LEVELCAL_MAX_TICKERS, LEVELCAL_START, LEVELCAL_END
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

START = os.environ.get("LEVELCAL_START", "2010-01-01")
END = os.environ.get("LEVELCAL_END", "2019-12-31")   # 2020+ sealed for out-of-sample
MAX_TICKERS = int(os.environ.get("LEVELCAL_MAX_TICKERS", "150"))

LOOKBACK = 500        # trailing bars used to construct zones
REBUILD_EVERY = 21    # rebuild cadence (~monthly)
SEARCH_HORIZON = 21   # bars after a rebuild in which a touch may occur
MAX_HOLD = 40         # bars over which the outcome is resolved
HOLD_REACTION_ATR = 1.0   # rally that counts as "held"
BREAK_CLOSE_ATR = 0.5     # close beyond the band that counts as "broken"
MIN_CELL_N = 200
RANDOM_SEED = 7


@dataclass
class Observation:
    ticker: str
    year: int
    strength: float
    regime: str
    held: bool
    bars_to_1r: float | None
    bars_to_2r: float | None
    break_cycles: int
    is_placebo: bool


def _resolve_outcome(
    bars: pd.DataFrame, atr_arr: np.ndarray, i: int, lo: float, hi: float
) -> tuple[bool, float | None, float | None] | None:
    """Forward-resolve one test of a band at bar `i`.

    `held` is a property of the zone and needs no entry price: from the band, did
    price rally >=1 ATR above it before *closing* >=0.5 ATR below it?

    The R-multiples are properties of a trade, so they are priced at what could
    actually be paid — the open of the bar after the touch bar, matching how the
    eventual strategy would fill.
    """
    n = len(bars)
    a = atr_arr[i]
    if not np.isfinite(a) or a <= 0 or i + 1 >= n:
        return None

    high = bars["high"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)

    end = min(i + 1 + MAX_HOLD, n)
    if end <= i + 1:
        return None

    entry = float(open_[i + 1])
    stop = lo - BREAK_CLOSE_ATR * a
    risk = entry - stop
    if risk <= 0:
        return None

    target_hold = hi + HOLD_REACTION_ATR * a
    held = False
    bars_to_1r: float | None = None
    bars_to_2r: float | None = None

    for j in range(i + 1, end):
        if bars_to_1r is None and high[j] >= entry + risk:
            bars_to_1r = float(j - i)
        if bars_to_2r is None and high[j] >= entry + 2.0 * risk:
            bars_to_2r = float(j - i)
        if high[j] >= target_hold:
            held = True
            break
        if close[j] < lo - BREAK_CLOSE_ATR * a:
            break   # broke before it ever rallied
    return held, bars_to_1r, bars_to_2r


def _first_touch(bars: pd.DataFrame, start: int, horizon: int, lo: float, hi: float) -> int | None:
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    end = min(start + horizon, len(bars))
    for i in range(start, end):
        if low[i] <= hi and high[i] >= lo:
            return i
    return None


def collect(ticker: str, bars: pd.DataFrame, rng: random.Random) -> list[Observation]:
    obs: list[Observation] = []
    if len(bars) < LOOKBACK + SEARCH_HORIZON + MAX_HOLD:
        return obs

    atr_arr = atr(bars, period=20).to_numpy(dtype=float)
    years = bars.index.year.to_numpy()

    for t in range(LOOKBACK, len(bars) - MAX_HOLD - SEARCH_HORIZON, REBUILD_EVERY):
        # Zones from PAST bars only. The builder cannot see bar t or later.
        window = bars.iloc[t - LOOKBACK : t]
        try:
            zones = build_zones(window)
        except Exception as exc:  # a single bad ticker must not kill the study
            logger.debug(f"{ticker}: zone build failed at {t}: {exc}")
            continue
        if not zones:
            continue

        last = len(window) - 1
        regime_snap = classify(window)
        regime = regime_snap.bias if regime_snap else "unknown"

        for zone in zones:
            strength = zone.strength_as_of(last)
            if strength <= 0.0:
                continue    # no sealed history yet; nothing to test
            i = _first_touch(bars, t, SEARCH_HORIZON, zone.lo, zone.hi)
            if i is None:
                continue
            resolved = _resolve_outcome(bars, atr_arr, i, zone.lo, zone.hi)
            if resolved is None:
                continue
            held, b1, b2 = resolved
            obs.append(Observation(
                ticker=ticker, year=int(years[i]), strength=strength, regime=regime,
                held=held, bars_to_1r=b1, bars_to_2r=b2,
                break_cycles=zone.break_cycles_as_of(last), is_placebo=False,
            ))

            # Matched placebo: same ticker, random date in the same year, a band
            # of identical width at that bar's low, with no structure behind it.
            p = _placebo_index(years, int(years[i]), rng)
            if p is not None and p + 1 + MAX_HOLD < len(bars):
                pa = atr_arr[p]
                if np.isfinite(pa) and pa > 0:
                    width = zone.hi - zone.lo
                    p_lo = float(bars["low"].iloc[p])
                    p_res = _resolve_outcome(bars, atr_arr, p, p_lo, p_lo + width)
                    if p_res is not None:
                        obs.append(Observation(
                            ticker=ticker, year=int(years[p]), strength=strength,
                            regime=regime, held=p_res[0], bars_to_1r=p_res[1],
                            bars_to_2r=p_res[2], break_cycles=0, is_placebo=True,
                        ))
    return obs


def _placebo_index(years: np.ndarray, year: int, rng: random.Random) -> int | None:
    idx = np.flatnonzero(years == year)
    if len(idx) < 10:
        return None
    return int(rng.choice(idx[:-MAX_HOLD] if len(idx) > MAX_HOLD else idx))


def _two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided p-value for a difference in proportions."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = s1 / n1, s2 / n2
    pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    # Normal two-sided tail without pulling in scipy
    return float(math.erfc(z / math.sqrt(2.0)))


def report(df: pd.DataFrame) -> None:
    real = df[~df.is_placebo].copy()
    placebo = df[df.is_placebo].copy()
    if real.empty:
        print("No observations collected.")
        return

    real["decile"] = pd.qcut(real["strength"].rank(method="first"), 10, labels=False) + 1

    print("\n" + "=" * 78)
    print("  LEVEL STRENGTH CALIBRATION — does support strength predict holding?")
    print("=" * 78)
    print(f"  Window {START} → {END}   tickers={real.ticker.nunique()}   "
          f"observations={len(real)}  placebo={len(placebo)}")
    print(f"  Zones built causally from a trailing {LOOKBACK}-bar window, "
          f"rebuilt every {REBUILD_EVERY} bars.\n")

    rows = []
    for d, g in real.groupby("decile"):
        rows.append({
            "decile": int(d), "n": len(g),
            "strength_lo": g.strength.min(), "strength_hi": g.strength.max(),
            "hold_rate": g.held.mean(),
            "med_bars_1R": g.bars_to_1r.median(),
            "med_bars_2R": g.bars_to_2r.median(),
            "reliable": "" if len(g) >= MIN_CELL_N else "  LOW-N",
        })
    table = pd.DataFrame(rows).set_index("decile")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    p_rate = placebo.held.mean() if not placebo.empty else float("nan")
    print(f"\n  Placebo (random dates, no structure): hold rate {p_rate:.3f}  n={len(placebo)}")
    print("  Absolute rates are inflated by survivorship bias — read the SPREAD, not the level.")

    bottom = real[real.decile <= 3]
    top = real[real.decile >= 8]
    spread = top.held.mean() - bottom.held.mean()
    pval = _two_proportion_z(int(top.held.sum()), len(top), int(bottom.held.sum()), len(bottom))
    rho = table["hold_rate"].corr(pd.Series(table.index, index=table.index), method="spearman")
    top_decile = real[real.decile == 10].held.mean()
    lift = top_decile - p_rate if not placebo.empty else float("nan")

    print("\n" + "-" * 78)
    print("  KILL CRITERIA")
    print("-" * 78)
    c1 = spread >= 0.06 and pval < 0.05
    c2 = rho >= 0.6
    c3 = lift >= 0.05
    print(f"  1. Tercile spread (D8-10 minus D1-3) >= 6pp, p<0.05 : "
          f"{spread*100:+.1f}pp  p={pval:.4f}   {'PASS' if c1 else 'FAIL'}")
    print(f"  2. Spearman rho(decile, hold rate) >= 0.60          : "
          f"{rho:+.3f}                {'PASS' if c2 else 'FAIL'}")
    print(f"  3. Top decile beats placebo by >= 5pp               : "
          f"{lift*100:+.1f}pp               {'PASS' if c3 else 'FAIL'}")

    if lift > 0.15:
        print("\n  !! Lift exceeds 15pp — treat as a look-ahead bug until proven otherwise.")

    print("\n  " + ("PROCEED — build the setup engine." if (c1 and c2 and c3)
                    else "STOP. Level strength does not predict holding. Record in DECISIONS.md."))

    if not real.break_cycles.eq(0).all():
        print("\n  Battleground cohort (zones broken and reclaimed):")
        bc = real.groupby(real.break_cycles.clip(0, 3)).agg(n=("held", "size"), hold=("held", "mean"))
        print(bc.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n  By regime:")
    print(real.groupby("regime").agg(n=("held", "size"), hold=("held", "mean"))
          .to_string(float_format=lambda v: f"{v:,.3f}"))


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    tickers = get_sp500_tickers()[:MAX_TICKERS]
    logger.info(f"Level calibration: {len(tickers)} tickers, {START} → {END}")

    bars = fetch_daily_ohlcv(tickers, START, END)
    bars = apply_size_filter(bars)

    all_obs: list[Observation] = []
    for n, (ticker, df) in enumerate(sorted(bars.items()), 1):
        all_obs.extend(collect(ticker, df, rng))
        if n % 25 == 0:
            logger.info(f"  {n}/{len(bars)} tickers, {len(all_obs)} observations")

    if not all_obs:
        print("No observations collected — check the universe and date range.")
        return
    report(pd.DataFrame([o.__dict__ for o in all_obs]))


if __name__ == "__main__":
    main()
