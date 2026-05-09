from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from signals.gap import GapSignal, StockSnapshot
from signals.orb import ORBSignal
from signals.vwap import VWAPSignal


@dataclass
class IntradayCandidate:
    """A stock selected in the morning scan as a trade candidate for the day."""

    ticker: str
    direction: str        # "long" or "short" — directional bias from gap (informational)
    gap_pct: float
    gap_score: float      # composite morning score (higher = stronger setup)
    prev_close: float
    atr: float
    orb_high: float | None = None
    orb_low: float | None = None


class IntradayComposite:
    """Morning scan + intraday signal hub — VWAP-mean-reversion primary.

    Phase 1 (09:25 ET): `rank_candidates()` — gap screen + volume/ATR liquidity
    filter, returns the top-N tickers worth fading on the day.

    Phase 2 (intraday): `on_bar()` — feeds bars to VWAP (primary entry) and
    ORB (range tracking only — no longer used for trend-following entries).
    Day-type sizing is applied by the caller via `get_day_type()`.
    """

    def __init__(
        self,
        top_n: int = 15,
        gap_min_pct: float = 1.5,
        volume_ratio_min: float = 2.0,
        atr_min_dollars: float = 0.50,
        orb_minutes: int = 15,
        vwap_std_threshold: float = 1.5,
        vwap_min_dollar_deviation_pct: float = 0.015,
        vwap_confirm_reversal: bool = False,
        orb_volume_confirm_ratio: float = 0.5,
    ):
        self._top_n = top_n
        self._gap = GapSignal(gap_min_pct, volume_ratio_min, atr_min_dollars)
        self._orb = ORBSignal(orb_minutes, volume_confirm_ratio=orb_volume_confirm_ratio)
        self._vwap = VWAPSignal(
            vwap_std_threshold,
            vwap_min_dollar_deviation_pct,
            confirm_reversal=vwap_confirm_reversal,
        )

        self._candidates: dict[str, IntradayCandidate] = {}
        self._day_type: str = "range"   # default favors VWAP — flip to "trend" only on confirmation

    def reset(self) -> None:
        """Call at 09:25 ET each morning before the scan."""
        self._candidates.clear()
        self._orb.reset()
        self._vwap.reset()
        self._day_type = "range"

    # ------------------------------------------------------------------
    # Phase 1: morning scan
    # ------------------------------------------------------------------

    def rank_candidates(self, snapshots: list[StockSnapshot]) -> list[IntradayCandidate]:
        """Run the gap screen and return the top-N trade candidates."""
        scores = self._gap.score(snapshots)
        snap_map = {s.ticker: s for s in snapshots}

        sorted_tickers = sorted(scores, key=lambda t: abs(scores[t]), reverse=True)
        candidates: list[IntradayCandidate] = []

        for ticker in sorted_tickers[: self._top_n]:
            snap = snap_map[ticker]
            score = scores[ticker]
            direction = "long" if score > 0 else "short"
            c = IntradayCandidate(
                ticker=ticker,
                direction=direction,
                gap_pct=snap.gap_pct,
                gap_score=abs(score),
                prev_close=snap.prev_close,
                atr=snap.atr,
            )
            candidates.append(c)
            self._candidates[ticker] = c

        logger.info(
            f"Morning scan: {len(candidates)} candidates — "
            + ", ".join(f"{c.ticker} ({c.direction}, gap={c.gap_pct:+.1%})" for c in candidates[:5])
            + ("…" if len(candidates) > 5 else "")
        )
        return candidates

    # ------------------------------------------------------------------
    # Phase 2: intraday signal routing
    # ------------------------------------------------------------------

    def set_day_type(self, day_type: str) -> None:
        """Called at 10:00 ET after SPY opening 30-min action is observed."""
        self._day_type = day_type
        logger.info(f"Day classified as: {day_type}")

    def get_day_type(self) -> str:
        return self._day_type

    def on_bar(self, ticker: str, bar: dict) -> str | None:
        """Process a 1-min bar for a candidate ticker.

        Returns "fade_long", "fade_short", or None.

        ORB is still updated (its range is informational and may be used as a
        sanity check) but no longer emits trend-following entries — VWAP mean
        reversion is the only entry source.
        """
        if ticker not in self._candidates:
            return None

        candidate = self._candidates[ticker]

        # Update ORB state for range tracking (no signal consumed)
        self._orb.on_bar(ticker, bar)
        orb_range = self._orb.get_range(ticker)
        if orb_range and candidate.orb_high is None:
            candidate.orb_high = orb_range.high
            candidate.orb_low = orb_range.low

        # VWAP fade — primary (and only) entry
        return self._vwap.on_bar(ticker, bar)

    def get_candidates(self) -> list[IntradayCandidate]:
        return list(self._candidates.values())

    def get_vwap(self, ticker: str) -> float | None:
        return self._vwap.get_vwap(ticker)

    def get_vwap_std(self, ticker: str) -> float | None:
        return self._vwap.get_std_dev(ticker)
