"""Composite-score math and threshold gate."""
from __future__ import annotations

import pytest

from signals.scoring import (
    DEFAULT_WEIGHTS,
    score_setup,
    passes_threshold,
)


class TestScoring:
    def test_max_score_with_all_components(self):
        card = score_setup(
            at_sr_level=True,
            candle_triggered=True,
            trend_aligned=True,
            rsi_aligned=True,
            macd_aligned=True,
            volume_confirmed=True,
        )
        assert card.total == sum(DEFAULT_WEIGHTS.values()) == 100

    def test_zero_score(self):
        card = score_setup(
            at_sr_level=False, candle_triggered=False, trend_aligned=False,
            rsi_aligned=False, macd_aligned=False, volume_confirmed=False,
        )
        assert card.total == 0

    def test_partial_score_components_correct(self):
        # S/R + candle + trend + volume = 25 + 20 + 20 + 10 = 75
        card = score_setup(
            at_sr_level=True, candle_triggered=True, trend_aligned=True,
            rsi_aligned=False, macd_aligned=False, volume_confirmed=True,
        )
        assert card.total == 75
        assert card.components == {
            "sr_level": 25, "candlestick": 20, "trend_alignment": 20,
            "rsi": 0, "macd": 0, "volume": 10,
        }

    def test_threshold_gate(self):
        below = score_setup(
            at_sr_level=True, candle_triggered=True, trend_aligned=False,
            rsi_aligned=True, macd_aligned=False, volume_confirmed=False,
        )  # 25 + 20 + 15 = 60
        above = score_setup(
            at_sr_level=True, candle_triggered=True, trend_aligned=True,
            rsi_aligned=False, macd_aligned=False, volume_confirmed=False,
        )  # 25 + 20 + 20 = 65
        assert not passes_threshold(below, threshold=65)
        assert passes_threshold(above, threshold=65)

    def test_custom_weights_override(self):
        card = score_setup(
            at_sr_level=True, candle_triggered=False, trend_aligned=False,
            rsi_aligned=False, macd_aligned=False, volume_confirmed=False,
            weights={"sr_level": 50},
        )
        assert card.components["sr_level"] == 50
        assert card.total == 50
