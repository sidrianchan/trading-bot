from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from loguru import logger


@dataclass
class OpeningRange:
    """First-N-minute high/low for a single stock."""

    ticker: str
    high: float
    low: float
    established_at: datetime
    bar_count: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def breakout_long(self, price: float, buffer_pct: float = 0.001) -> bool:
        return price > self.high * (1 + buffer_pct)

    def breakout_short(self, price: float, buffer_pct: float = 0.001) -> bool:
        return price < self.low * (1 - buffer_pct)


class ORBSignal:
    """Opening Range Breakout detector on streaming 1-minute bars.

    Tracks the high/low of the first `orb_minutes` minutes after open for
    each stock. After the range is established, emits a breakout signal when
    a bar closes above range high (long) or below range low (short).

    Designed to run in the agent loop: call `on_bar()` for each incoming bar.
    """

    def __init__(self, orb_minutes: int = 15, volume_confirm_ratio: float = 1.2):
        self.orb_minutes = orb_minutes
        self.volume_confirm_ratio = volume_confirm_ratio

        self._bars: dict[str, list[dict]] = {}         # ticker → list of bar dicts
        self._ranges: dict[str, OpeningRange] = {}     # ticker → established range
        self._avg_bar_volume: dict[str, float] = {}    # ticker → recent avg bar vol

    def reset(self) -> None:
        """Call at the start of each trading day."""
        self._bars.clear()
        self._ranges.clear()
        self._avg_bar_volume.clear()

    def on_bar(self, ticker: str, bar: dict) -> str | None:
        """Process one 1-min bar.

        Returns:
            "long"  — confirmed breakout above ORB high
            "short" — confirmed breakout below ORB low
            None    — no signal yet
        """
        if ticker not in self._bars:
            self._bars[ticker] = []

        self._bars[ticker].append(bar)
        bars = self._bars[ticker]

        if ticker not in self._ranges:
            if len(bars) >= self.orb_minutes:
                self._ranges[ticker] = self._establish_range(ticker, bars[:self.orb_minutes])
                # Store average bar volume from ORB period for confirmation
                self._avg_bar_volume[ticker] = sum(b["volume"] for b in bars[:self.orb_minutes]) / self.orb_minutes
            return None

        orb = self._ranges[ticker]
        close = bar["close"]
        volume = bar["volume"]
        avg_vol = self._avg_bar_volume.get(ticker, 1)

        volume_confirmed = volume >= avg_vol * self.volume_confirm_ratio

        if volume_confirmed and orb.breakout_long(close):
            logger.info(f"{ticker}: ORB long breakout @ {close:.2f} (range {orb.low:.2f}–{orb.high:.2f})")
            return "long"

        if volume_confirmed and orb.breakout_short(close):
            logger.info(f"{ticker}: ORB short breakout @ {close:.2f} (range {orb.low:.2f}–{orb.high:.2f})")
            return "short"

        return None

    def get_range(self, ticker: str) -> OpeningRange | None:
        return self._ranges.get(ticker)

    def _establish_range(self, ticker: str, bars: list[dict]) -> OpeningRange:
        high = max(b["high"] for b in bars)
        low = min(b["low"] for b in bars)
        orb = OpeningRange(
            ticker=ticker,
            high=high,
            low=low,
            established_at=datetime.now(),
            bar_count=len(bars),
        )
        logger.info(f"{ticker}: ORB established {low:.2f}–{high:.2f} (width ${high - low:.2f})")
        return orb
