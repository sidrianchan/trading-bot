"""Composite scoring (0-100) for technical-analysis setups.

Scoring components (matches the spec):

| Component             | Max points |
|-----------------------|------------|
| S/R level             | 25         |
| Candlestick pattern   | 20         |
| Trend alignment       | 20         |
| RSI confirmation      | 15         |
| MACD confirmation     | 10         |
| Volume confirmation   | 10         |
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_WEIGHTS: dict[str, int] = {
    "sr_level": 25,
    "candlestick": 20,
    "trend_alignment": 20,
    "rsi": 15,
    "macd": 10,
    "volume": 10,
}


@dataclass(frozen=True)
class ScoreCard:
    components: dict[str, float]
    total: float

    @property
    def passes(self) -> bool:
        return False  # threshold check belongs to the caller; see ``score_setup``


def score_setup(
    *,
    at_sr_level: bool,
    candle_triggered: bool,
    trend_aligned: bool,
    rsi_aligned: bool,
    macd_aligned: bool,
    volume_confirmed: bool,
    weights: dict[str, int] | None = None,
) -> ScoreCard:
    """Compute the composite 0-100 score from boolean component checks.

    Each component is fully on (max points) or fully off (0 points). Partial
    credit is reserved for the future — start strict so the gate signal is
    crisp.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components: dict[str, float] = {
        "sr_level":         w["sr_level"]        if at_sr_level     else 0,
        "candlestick":      w["candlestick"]     if candle_triggered else 0,
        "trend_alignment":  w["trend_alignment"] if trend_aligned   else 0,
        "rsi":              w["rsi"]             if rsi_aligned     else 0,
        "macd":             w["macd"]            if macd_aligned    else 0,
        "volume":           w["volume"]          if volume_confirmed else 0,
    }
    return ScoreCard(components=components, total=float(sum(components.values())))


def passes_threshold(card: ScoreCard, threshold: float = 65.0) -> bool:
    return card.total >= threshold
