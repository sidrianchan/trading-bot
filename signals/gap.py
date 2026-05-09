from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class StockSnapshot:
    """Pre-market snapshot used for the 09:25 morning scan."""

    ticker: str
    prev_close: float
    latest_price: float        # most recent pre-market or last trade price
    pre_market_volume: int
    avg_volume_30d: float      # 30-day average daily volume
    atr: float                 # 14-day ATR in dollars
    gap_pct: float             # (latest_price / prev_close) - 1
    volume_ratio: float        # pre_market_volume / (avg_volume_30d × 0.25)


class GapSignal:
    """Morning gap screen: ranks stocks by gap magnitude × volume confirmation.

    Scores are directional — positive = gap-up candidate (long),
    negative = gap-down candidate (short). Only stocks meeting the
    minimum gap, volume, and ATR thresholds produce a non-zero score.
    """

    def __init__(
        self,
        gap_min_pct: float = 1.5,
        volume_ratio_min: float = 2.0,
        atr_min_dollars: float = 0.50,
    ):
        self.gap_min_pct = gap_min_pct / 100.0
        self.volume_ratio_min = volume_ratio_min
        self.atr_min_dollars = atr_min_dollars

    def score(self, snapshots: list[StockSnapshot]) -> dict[str, float]:
        """Return ticker → score map. Score is 0 for disqualified stocks."""
        scores: dict[str, float] = {}
        qualified = 0

        for snap in snapshots:
            if snap.atr < self.atr_min_dollars:
                continue
            if abs(snap.gap_pct) < self.gap_min_pct:
                continue
            if snap.volume_ratio < self.volume_ratio_min:
                continue

            # Score = gap magnitude × volume confirmation (clipped at 10×)
            vol_factor = min(snap.volume_ratio / self.volume_ratio_min, 10.0)
            raw = abs(snap.gap_pct) * vol_factor
            scores[snap.ticker] = raw if snap.gap_pct > 0 else -raw
            qualified += 1

        logger.info(
            f"Gap screen: {len(snapshots)} stocks → {qualified} qualified "
            f"(gap ≥ {self.gap_min_pct:.1%}, vol ratio ≥ {self.volume_ratio_min}×)"
        )
        return scores
