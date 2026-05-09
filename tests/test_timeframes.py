"""Resampling and market-hours filters in data/timeframes.py."""
from __future__ import annotations

import pandas as pd
import pytest

from data.timeframes import resample, market_hours_only


def _make_1min(periods: int = 90, start: str = "2024-01-02 09:30", tz: str | None = "America/New_York") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1min", tz=tz)
    return pd.DataFrame(
        {
            "open":  [100.0 + i * 0.01 for i in range(periods)],
            "high":  [100.5 + i * 0.01 for i in range(periods)],
            "low":   [ 99.5 + i * 0.01 for i in range(periods)],
            "close": [100.2 + i * 0.01 for i in range(periods)],
            "volume": [1_000 + i for i in range(periods)],
        },
        index=idx,
    )


class TestResample:
    def test_15min_aggregates_correctly(self):
        bars = _make_1min(60)        # 1 hour of data => 4 × 15-min bars
        out = resample(bars, "15min")
        assert len(out) == 4
        # First 15-min bar = first 15 1-min bars (right-edge labelled at 09:45)
        first = out.iloc[0]
        src = bars.iloc[:15]
        assert first["open"] == src["open"].iloc[0]
        assert first["high"] == src["high"].max()
        assert first["low"] == src["low"].min()
        assert first["close"] == src["close"].iloc[-1]
        assert first["volume"] == src["volume"].sum()

    def test_1h_aggregates_correctly(self):
        bars = _make_1min(120)       # 2 hours => 2 × 1H bars
        out = resample(bars, "1h")
        assert len(out) == 2

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index = pd.DatetimeIndex([], tz="America/New_York")
        out = resample(empty, "15min")
        assert out.empty

    def test_missing_columns_raises(self):
        bad = pd.DataFrame({"open": [1.0], "close": [1.0]})
        bad.index = pd.DatetimeIndex(["2024-01-02 09:30"], tz="America/New_York")
        with pytest.raises(ValueError, match="missing columns"):
            resample(bad, "15min")


class TestMarketHoursOnly:
    def test_keeps_session_strips_premarket(self):
        # 60 minutes of pre-market + 30 minutes of session
        idx = pd.date_range("2024-01-02 08:30", periods=90, freq="1min", tz="America/New_York")
        df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx)
        out = market_hours_only(df)
        assert out.index.min() >= pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
        assert out.index.max() <= pd.Timestamp("2024-01-02 16:00", tz="America/New_York")

    def test_handles_utc_index(self):
        idx = pd.date_range("2024-01-02 14:30", periods=30, freq="1min", tz="UTC")  # = 09:30 ET
        df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
        out = market_hours_only(df)
        assert len(out) == 30
