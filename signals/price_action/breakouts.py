"""Breakout pattern detectors.

Two flavors:

* **consolidation breakout** — price compresses (ATR contraction), then a
  bar closes outside the prior range with above-average volume.
* **flag / pennant continuation** — strong impulse leg, brief sideways or
  counter-trend consolidation, break in the impulse direction.

Both detectors are conservative — they only trigger on the most recent bar
(no historical re-triggering) so the backtester can call them per-bar without
inflating signal counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from signals.indicators.volatility import atr as atr_series
from signals.indicators.volume import volume_ratio

BreakoutKind = Literal["consolidation", "flag"]
Direction = Literal["long", "short"]


@dataclass(frozen=True)
class BreakoutSignal:
    triggered: bool
    kind: BreakoutKind | None = None
    direction: Direction | None = None
    breakout_level: float = 0.0
    volume_mult: float = 0.0
    detail: str = ""


def consolidation_breakout(
    bars: pd.DataFrame,
    *,
    consolidation_window: int = 10,
    atr_short: int = 14,
    atr_long: int = 50,
    atr_contraction: float = 0.7,
    min_volume_multiple: float = 1.5,
) -> BreakoutSignal:
    """Detect a volume-confirmed breakout from a tight consolidation.

    Conditions on the latest bar:

    * ATR(short) / ATR(long) computed up to the *prior* bar ≤ ``atr_contraction``
      (price has been compressing).
    * Latest bar's close is strictly above (long) or below (short) the
      ``consolidation_window`` prior bars' range.
    * Latest bar's volume ≥ ``min_volume_multiple`` × the average of the
      prior 20 bars.
    """
    if len(bars) < max(atr_long, consolidation_window) + 2:
        return BreakoutSignal(False, detail="insufficient history")

    prior = bars.iloc[:-1]
    short_atr = atr_series(prior, period=atr_short)
    long_atr = atr_series(prior, period=atr_long)
    if short_atr.dropna().empty or long_atr.dropna().empty:
        return BreakoutSignal(False, detail="ATR not yet warm")
    a_short = float(short_atr.iloc[-1])
    a_long = float(long_atr.iloc[-1])
    if a_long <= 0 or (a_short / a_long) > atr_contraction:
        return BreakoutSignal(False, detail=f"no ATR contraction ({a_short / max(a_long, 1e-9):.2f})")

    window = bars.iloc[-consolidation_window - 1 : -1]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    last = bars.iloc[-1]
    last_close = float(last["close"])
    vmult = volume_ratio(bars, period=20)

    if vmult < min_volume_multiple:
        return BreakoutSignal(False, detail=f"volume {vmult:.2f}× < {min_volume_multiple}×")

    if last_close > range_high:
        return BreakoutSignal(
            True, "consolidation", "long", range_high, vmult,
            f"close {last_close:.2f} > range high {range_high:.2f}",
        )
    if last_close < range_low:
        return BreakoutSignal(
            True, "consolidation", "short", range_low, vmult,
            f"close {last_close:.2f} < range low {range_low:.2f}",
        )
    return BreakoutSignal(False, detail="inside range")


def flag_breakout(
    bars: pd.DataFrame,
    *,
    impulse_window: int = 5,
    flag_window: int = 5,
    impulse_min_pct: float = 0.05,
    min_volume_multiple: float = 1.2,
) -> BreakoutSignal:
    """Flag/pennant continuation breakout.

    Pattern, indexed from the latest bar working backward:

    * ``flag_window`` bars form a tight pullback.
    * The ``impulse_window`` bars BEFORE the flag form a strong impulse
      (≥ ``impulse_min_pct`` move in one direction).
    * Latest bar closes beyond the flag's high (long) / low (short) on
      above-average volume.
    """
    needed = impulse_window + flag_window + 1
    if len(bars) < needed:
        return BreakoutSignal(False, detail="insufficient history")

    last = bars.iloc[-1]
    flag = bars.iloc[-flag_window - 1 : -1]
    impulse = bars.iloc[-flag_window - impulse_window - 1 : -flag_window - 1]
    if flag.empty or impulse.empty:
        return BreakoutSignal(False, detail="window empty")

    impulse_start = float(impulse["close"].iloc[0])
    impulse_end = float(impulse["close"].iloc[-1])
    if impulse_start <= 0:
        return BreakoutSignal(False, detail="impulse start <= 0")
    move = (impulse_end - impulse_start) / impulse_start

    flag_high = float(flag["high"].max())
    flag_low = float(flag["low"].min())
    last_close = float(last["close"])
    vmult = volume_ratio(bars, period=20)

    if vmult < min_volume_multiple:
        return BreakoutSignal(False, detail=f"flag vol {vmult:.2f}× < {min_volume_multiple}×")

    if move >= impulse_min_pct and last_close > flag_high:
        return BreakoutSignal(
            True, "flag", "long", flag_high, vmult,
            f"impulse +{move:.1%}, close {last_close:.2f} > flag high {flag_high:.2f}",
        )
    if move <= -impulse_min_pct and last_close < flag_low:
        return BreakoutSignal(
            True, "flag", "short", flag_low, vmult,
            f"impulse {move:.1%}, close {last_close:.2f} < flag low {flag_low:.2f}",
        )
    return BreakoutSignal(False, detail="no continuation trigger")
