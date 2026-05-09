"""Multi-timeframe resampling helpers.

Phase A scaffolding — actual implementation lands in Phase B alongside the
detectors that consume it. Defines the public API up-front so the new
``signals/`` modules can import these symbols without circular dependencies.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

Timeframe = Literal["15T", "1H", "1D"]


def resample(bars_1min: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to a higher timeframe.

    Phase A: not yet implemented. Phase B fills this in (right-edge labeled
    bars, drop incomplete trailing bar, preserve ET timezone awareness).
    """
    raise NotImplementedError("data.timeframes.resample lands in Phase B")


def market_hours_only(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter a bar series to regular US equity market hours (09:30-16:00 ET)."""
    raise NotImplementedError("data.timeframes.market_hours_only lands in Phase B")
