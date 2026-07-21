"""Do pullbacks to the 20-EMA in a strong uptrend bounce, or is it just drift?

A companion probe to `scripts/level_calibration.py` (static horizontal support) and
`scripts/breakdown_calibration.py` (breakdowns of that support). Those measured
*static* levels; this one pivots to a *dynamic* mean. The hypothesis: in a strict
structural uptrend, a pullback that touches the rising 20-day EMA reverts a full
1R more often than chance — and does so *more* the more stretched the trend was
before the pullback (trend elasticity: a rubber band snaps back harder the further
you pull it).

This is still a measurement study. No backtester, no setup engine, no scoring. It
asks one question per Stretch tercile: of the pullbacks that touched the mean, what
fraction reached +1 ATR (target) before -1 ATR (stop)?

THE PLACEBO — controlling for upside drift (the Ghost Mean)
-----------------------------------------------------------
2010–2019 drifted strongly upward, which *inflates* any long-side bounce rate
uniformly — the same reason the hold-side study's absolute hold rate was
uninterpretable. So an absolute bounce rate is meaningless on its own. For every
trend-leg we build a structureless "Ghost Mean": the 20-EMA shifted by a static
random offset (0.5–1.5 ATR, either side, avoiding the ±0.2 ATR dead-zone around the
true mean), chosen once for that leg. We wait for price to touch the ghost under the
*identical* regime rules and resolve the same symmetric 1R. The ghost has no
mean-reversion pull behind it, so its bounce rate is the base rate of "buy any
random dip in this name, at this time." The signal is the *difference*.

LOOK-AHEAD SEALING
------------------
1. **The trigger line is the 20-EMA as of t-1**, never t. A touch on day t is
   evaluated against `ema20[t-1]`, so the touch cannot move the line it touched.
   The EMA is causal (`adjust=False`), so `ema20[t-1]` is bit-identical whether or
   not any bar >= t exists — proven by `test_dynamic_seal_does_not_leak`.
2. **Stretch is accumulated only over bars <= t-1** (the leg that ended at the
   touch), using each bar's own sealed EMA/ATR.
3. **Risk is sized from ATR at t-1**, never the touch bar's own (often large) range.
4. **Entry is the open of t+1**, never the touch-day close — the first price a long
   could actually be filled at after the signal prints.
5. The regime stack (20>50>200 for >=10 bars) is evaluated at t-1.

Run: .venv/bin/python3 scripts/trend_pullback_calibration.py
Env: TRENDCAL_MAX_TICKERS, TRENDCAL_START, TRENDCAL_END
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
from signals.indicators.trend import ema
from signals.indicators.volatility import atr

START = os.environ.get("TRENDCAL_START", "2010-01-01")
END = os.environ.get("TRENDCAL_END", "2019-12-31")   # 2020+ sealed for out-of-sample
MAX_TICKERS = int(os.environ.get("TRENDCAL_MAX_TICKERS", "150"))

EMA_FAST = 20
EMA_SLOW = 50
SMA_LONG = 200          # a 200-EMA proxy for the long structural filter
ATR_PERIOD = 20
REGIME_BARS = 10        # bars the 20>50>200 stack must hold, as of t-1
MAX_HOLD = 40           # bars over which the 1R outcome is resolved
GHOST_HORIZON = 60      # bars a ghost mean gets to be touched before we give up

GHOST_MIN_OFFSET = 0.5  # ghost placed |0.5..1.5| ATR from the true mean...
GHOST_MAX_OFFSET = 1.5  # ...on either side (the ±0.2 dead-zone is thus avoided).

MIN_CELL_N = 200
RANDOM_SEED = 7


@dataclass
class PullbackObs:
    ticker: str
    year: int
    stretch: float
    outcome: str          # "win" | "loss" | "neither"
    bars_to_1r: float | None
    is_placebo: bool


def trigger_lines(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The causal indicator lines the probe reads, as numpy arrays.

    Returns ``(ema20, ema50, ema200, atr20)`` aligned to ``bars``. All are causal:
    the value at index i depends only on bars <= i, so the value at t-1 is sealed
    against anything that happens on day t. Factored out so the seal is unit-testable.
    """
    close = bars["close"]
    ema20 = ema(close, EMA_FAST).to_numpy(dtype=float)
    ema50 = ema(close, EMA_SLOW).to_numpy(dtype=float)
    ema200 = ema(close, SMA_LONG).to_numpy(dtype=float)
    atr20 = atr(bars, period=ATR_PERIOD).to_numpy(dtype=float)
    return ema20, ema50, ema200, atr20


def _regime_run(ema20: np.ndarray, ema50: np.ndarray, ema200: np.ndarray) -> np.ndarray:
    """Consecutive-bar count of the strict 20>50>200 uptrend stack, per bar.

    ``run[i]`` is how many consecutive bars up to and including ``i`` satisfy the
    stack. ``regime_ok(t)`` is then ``run[t-1] >= REGIME_BARS`` — the stack must have
    held for REGIME_BARS bars as of t-1.
    """
    stack = (ema20 > ema50) & (ema50 > ema200)
    stack &= np.isfinite(ema20) & np.isfinite(ema50) & np.isfinite(ema200)
    run = np.zeros(len(stack), dtype=int)
    c = 0
    for i, up in enumerate(stack):
        c = c + 1 if up else 0
        run[i] = c
    return run


def _is_touch(low: np.ndarray, high: np.ndarray, ema20: np.ndarray, b: int) -> bool:
    """Day b touches the mean if its range straddles the *t-1* EMA line."""
    if b < 1:
        return False
    line = ema20[b - 1]
    if not np.isfinite(line):
        return False
    return low[b] <= line <= high[b]


def _leg_max_stretch(
    high: np.ndarray, ema20: np.ndarray, atr_arr: np.ndarray, start: int, end: int,
) -> float:
    """Max ATR-distance the high reached above the mean over bars [start, end).

    This is the leg's "stretch" — how far the rubber band was pulled since the last
    touch. Called with ``end = t`` (the touch bar), so every bar read is <= t-1 and
    the signal is sealed against the day-t trigger. Uses each bar's own EMA/ATR.
    """
    best = 0.0
    for b in range(max(start, 0), end):
        a = atr_arr[b]
        if not (np.isfinite(a) and a > 0 and np.isfinite(ema20[b])):
            continue
        s = (high[b] - ema20[b]) / a
        if s > best:
            best = s
    return best


def _resolve_bounce(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr_arr: np.ndarray, t: int,
) -> tuple[str, float | None] | None:
    """Forward-resolve a symmetric 1R long from a touch at bar t.

    All sealed relative to the touch bar t:
      entry  = open[t+1]                 (first fillable price after the signal)
      target = entry + 1 * ATR(t-1)      (1R up)
      stop   = entry - 1 * ATR(t-1)      (1R down)

    A *win* reaches the target (intraday high) before the stop (intraday low). When a
    single bar's range spans both, we assume the stop filled first (conservative —
    a daily bar cannot resolve intrabar order, and we never want to overstate the
    edge). Everything that resolves to neither inside MAX_HOLD is "neither".
    """
    n = len(open_)
    if t + 1 >= n or t < 1:
        return None
    a = atr_arr[t - 1]
    if not np.isfinite(a) or a <= 0:
        return None

    entry = float(open_[t + 1])
    target = entry + a
    stop = entry - a

    end = min(t + 1 + MAX_HOLD, n)
    for j in range(t + 1, end):
        hit_target = high[j] >= target
        hit_stop = low[j] <= stop
        if hit_stop:                      # tie (target+stop same bar) resolves to stop
            return "loss", None
        if hit_target:
            return "win", float(j - t)
    return "neither", None


def _ghost_touch(
    low: np.ndarray, high: np.ndarray, ema20: np.ndarray, atr_arr: np.ndarray,
    run: np.ndarray, start: int, horizon: int, offset: float,
) -> int | None:
    """First bar in [start, start+horizon) that touches the ghost mean under regime.

    The ghost mean is the 20-EMA shifted by ``offset`` ATRs (contemporaneous ATR,
    so the band rides parallel to the true mean). Touch uses the t-1 ghost value and
    requires the same strict-uptrend regime, exactly like a real touch.
    """
    n = len(low)
    end = min(start + horizon, n)
    for g in range(start, end):
        if g < 1 or run[g - 1] < REGIME_BARS:
            continue
        line = ema20[g - 1]
        a = atr_arr[g - 1]
        if not (np.isfinite(line) and np.isfinite(a) and a > 0):
            continue
        ghost = line + offset * a
        if low[g] <= ghost <= high[g]:
            return g
    return None


def collect(ticker: str, bars: pd.DataFrame, rng: random.Random) -> list[PullbackObs]:
    obs: list[PullbackObs] = []
    if len(bars) < SMA_LONG + REGIME_BARS + MAX_HOLD + 2:
        return obs

    ema20, ema50, ema200, atr_arr = trigger_lines(bars)
    run = _regime_run(ema20, ema50, ema200)
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    years = bars.index.year.to_numpy()

    n = len(bars)
    # Start once the 200-EMA and a full regime window can exist; leave MAX_HOLD+1
    # bars of forward room so every trigger can be resolved.
    first = SMA_LONG + 1
    last = n - MAX_HOLD - 2

    leg_started = False           # have we seen a first touch to anchor a leg?
    leg_start = first
    leg_offset = 0.0

    def _draw_offset() -> float:
        mag = rng.uniform(GHOST_MIN_OFFSET, GHOST_MAX_OFFSET)
        return mag if rng.random() < 0.5 else -mag

    for t in range(first, last):
        if not _is_touch(low, high, ema20, t):
            continue

        # A touch closes the current leg. If a prior touch anchored the leg and the
        # regime holds at t-1, this is a valid trigger: measure the real 1R and a
        # matched ghost. Stretch is the leg's max excursion over [leg_start+1, t-1].
        if leg_started and run[t - 1] >= REGIME_BARS:
            resolved = _resolve_bounce(open_, high, low, atr_arr, t)
            if resolved is not None:
                stretch = _leg_max_stretch(high, ema20, atr_arr, leg_start + 1, t)
                obs.append(PullbackObs(
                    ticker=ticker, year=int(years[t]), stretch=stretch,
                    outcome=resolved[0], bars_to_1r=resolved[1], is_placebo=False,
                ))
                g = _ghost_touch(low, high, ema20, atr_arr, run,
                                 leg_start + 1, GHOST_HORIZON, leg_offset)
                if g is not None:
                    g_res = _resolve_bounce(open_, high, low, atr_arr, g)
                    if g_res is not None:
                        obs.append(PullbackObs(
                            ticker=ticker, year=int(years[g]), stretch=stretch,
                            outcome=g_res[0], bars_to_1r=g_res[1], is_placebo=True,
                        ))

        # Reset the leg regardless of whether the touch was a valid trigger.
        leg_started = True
        leg_start = t
        leg_offset = _draw_offset()

    return obs


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
        print("No pullback triggers collected.")
        return

    # Stretch terciles from the real triggers; the same edges bin the ghosts.
    real["tercile"], edges = pd.qcut(
        real["stretch"].rank(method="first"), 3,
        labels=["Low", "Mid", "High"], retbins=True,
    )
    real["win"] = real.outcome == "win"
    if not placebo.empty:
        placebo = placebo.merge(
            real[["ticker", "stretch", "tercile"]].drop_duplicates(
                subset=["ticker", "stretch"]),
            on=["ticker", "stretch"], how="left",
        )
        placebo["win"] = placebo.outcome == "win"

    print("\n" + "=" * 84)
    print("  TREND-PULLBACK CALIBRATION — do 20-EMA pullbacks in strong uptrends bounce?")
    print("=" * 84)
    print(f"  Window {START} → {END}   tickers={real.ticker.nunique()}   "
          f"touches={len(real)}  ghosts={len(placebo)}")
    print(f"  Strict uptrend: 20>50>200 EMA for >={REGIME_BARS} bars at t-1. "
          f"Symmetric 1R = ±1 ATR{ATR_PERIOD}(t-1).\n")

    order = ["Low", "Mid", "High"]
    rows = []
    for terc in order:
        g = real[real.tercile == terc]
        pg = placebo[placebo.tercile == terc] if not placebo.empty else placebo
        rows.append({
            "tercile": terc,
            "n_touches": len(g),
            "stretch_lo": g.stretch.min(),
            "stretch_hi": g.stretch.max(),
            "bounce_rate": g.win.mean(),
            "placebo_bounce_rate": pg.win.mean() if len(pg) else float("nan"),
            "median_bars_to_1R": g.loc[g.win, "bars_to_1r"].median(),
            "placebo_bars_to_1R": pg.loc[pg.win, "bars_to_1r"].median() if len(pg) else float("nan"),
            "reliable": "" if len(g) >= MIN_CELL_N else "  LOW-N",
        })
    table = pd.DataFrame(rows).set_index("tercile")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    # ── Overall aggregates ──
    real_rate = real.win.mean()
    placebo_rate = placebo.win.mean() if not placebo.empty else float("nan")
    real_med = real.loc[real.win, "bars_to_1r"].median()
    placebo_med = placebo.loc[placebo.win, "bars_to_1r"].median() if not placebo.empty else float("nan")

    print(f"\n  Overall bounce {real_rate:.3f}  vs  ghost {placebo_rate:.3f}   "
          f"(n={len(real)}, ghosts={len(placebo)})")
    print("  Upside drift inflates every rate uniformly — read the LIFT and the GRADIENT.")

    hi = real[real.tercile == "High"]
    lo = real[real.tercile == "Low"]
    gradient = hi.win.mean() - lo.win.mean()
    grad_p = _two_proportion_z(int(hi.win.sum()), len(hi), int(lo.win.sum()), len(lo))
    lift = real_rate - placebo_rate if not placebo.empty else float("nan")

    print("\n" + "-" * 84)
    print("  KILL CRITERIA")
    print("-" * 84)
    c1 = (not math.isnan(lift)) and lift >= 0.05
    c2 = gradient >= 0.05 and grad_p < 0.05
    c3 = (not math.isnan(real_med)) and (not math.isnan(placebo_med)) and real_med < placebo_med
    print(f"  1. Precision Lift: bounce(Overall) - ghost(Overall) >= 5pp   : "
          f"{lift*100:+.1f}pp                {'PASS' if c1 else 'FAIL'}")
    print(f"  2. Elasticity Gradient: bounce(High) - bounce(Low) >= 5pp,   : "
          f"{gradient*100:+.1f}pp  p={grad_p:.4f}   {'PASS' if c2 else 'FAIL'}")
    print("     p<0.05")
    print(f"  3. Momentum: median_bars_1R(Overall) < ghost_bars_1R        : "
          f"{real_med:.1f} vs {placebo_med:.1f} bars        {'PASS' if c3 else 'FAIL'}")

    if not math.isnan(lift) and lift > 0.15:
        print("\n  !! Lift exceeds 15pp — treat as a look-ahead bug until proven otherwise.")

    print("\n  " + ("PROCEED — the trend-elasticity edge is real. Design the setup engine."
                    if (c1 and c2 and c3)
                    else "STOP. Record the outcome in DECISIONS.md."))


def main() -> None:
    if END > "2019-12-31":
        raise SystemExit(
            f"TRENDCAL_END={END} breaks the seal — 2020+ is strictly out-of-sample.")

    rng = random.Random(RANDOM_SEED)
    tickers = get_sp500_tickers()[:MAX_TICKERS]
    logger.info(f"Trend-pullback calibration: {len(tickers)} tickers, {START} → {END}")

    bars = fetch_daily_ohlcv(tickers, START, END)
    bars = apply_size_filter(bars)

    all_obs: list[PullbackObs] = []
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
