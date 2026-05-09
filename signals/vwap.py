from __future__ import annotations

from loguru import logger


class VWAPSignal:
    """Running VWAP + standard deviation bands for intraday mean reversion.

    Accumulates bar-by-bar from 09:30. Emits a fade signal when price
    deviates more than `std_threshold` standard deviations from VWAP **and**
    the absolute price gap from VWAP exceeds `min_dollar_deviation_pct`
    (filters out noise on low-priced slow movers).

    Used as the primary entry signal in the VWAP-mean-reversion strategy:
      fade_long  → buy below VWAP, target VWAP, stop at vwap - stop_std*σ
      fade_short → sell above VWAP, target VWAP, stop at vwap + stop_std*σ
    """

    def __init__(
        self,
        std_threshold: float = 1.5,
        min_dollar_deviation_pct: float = 0.015,
        confirm_reversal: bool = False,
    ):
        self.std_threshold = std_threshold
        self.min_dollar_deviation_pct = min_dollar_deviation_pct
        self.confirm_reversal = confirm_reversal
        self._state: dict[str, _VWAPState] = {}

    def reset(self) -> None:
        """Call at market open each day."""
        self._state.clear()

    def on_bar(self, ticker: str, bar: dict) -> str | None:
        """Update VWAP state and check for reversion signal.

        Returns:
            "fade_long"  — price is far below VWAP, fade the drop (buy)
            "fade_short" — price is far above VWAP, fade the rally (sell short)
            None         — no signal (insufficient bars, no σ deviation, or
                           dollar deviation below min_dollar_deviation_pct)
        """
        if ticker not in self._state:
            self._state[ticker] = _VWAPState()

        state = self._state[ticker]
        state.update(bar)

        if state.n_bars < 5:
            return None

        vwap = state.vwap
        std = state.std_dev
        close = bar["close"]

        if std <= 0 or close <= 0 or vwap <= 0:
            return None

        deviation_sigma = (close - vwap) / std
        deviation_pct = abs(close - vwap) / close

        if deviation_pct < self.min_dollar_deviation_pct:
            return None

        prev_close = state.prev_close

        if deviation_sigma <= -self.std_threshold:
            # fade_long: only enter if last bar already turned back UP toward VWAP
            if self.confirm_reversal and (prev_close is None or close <= prev_close):
                return None
            logger.debug(
                f"{ticker}: VWAP fade-long (dev={deviation_sigma:.2f}σ, "
                f"pct={deviation_pct:.2%}, VWAP={vwap:.2f})"
            )
            return "fade_long"

        if deviation_sigma >= self.std_threshold:
            # fade_short: only enter if last bar already turned back DOWN toward VWAP
            if self.confirm_reversal and (prev_close is None or close >= prev_close):
                return None
            logger.debug(
                f"{ticker}: VWAP fade-short (dev={deviation_sigma:.2f}σ, "
                f"pct={deviation_pct:.2%}, VWAP={vwap:.2f})"
            )
            return "fade_short"

        return None

    def get_vwap(self, ticker: str) -> float | None:
        state = self._state.get(ticker)
        return state.vwap if state else None

    def get_std_dev(self, ticker: str) -> float | None:
        state = self._state.get(ticker)
        return state.std_dev if state else None

    def get_deviation(self, ticker: str, price: float) -> float | None:
        state = self._state.get(ticker)
        if state is None or state.std_dev <= 0:
            return None
        return (price - state.vwap) / state.std_dev


class _VWAPState:
    """Per-ticker rolling VWAP and variance computation."""

    def __init__(self) -> None:
        self._cum_tp_vol: float = 0.0   # Σ(typical_price × volume)
        self._cum_vol: float = 0.0      # Σ(volume)
        self._cum_tp2_vol: float = 0.0  # Σ(typical_price² × volume) for variance
        self.n_bars: int = 0
        self.prev_close: float | None = None
        self._last_close: float | None = None

    def update(self, bar: dict) -> None:
        tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        vol = float(bar["volume"])
        if vol <= 0:
            return
        # roll the previous bar's close before recording the current one
        self.prev_close = self._last_close
        self._last_close = float(bar["close"])
        self._cum_tp_vol += tp * vol
        self._cum_tp2_vol += tp * tp * vol
        self._cum_vol += vol
        self.n_bars += 1

    @property
    def vwap(self) -> float:
        if self._cum_vol <= 0:
            return 0.0
        return self._cum_tp_vol / self._cum_vol

    @property
    def std_dev(self) -> float:
        if self._cum_vol <= 0 or self.n_bars < 2:
            return 0.0
        variance = (self._cum_tp2_vol / self._cum_vol) - self.vwap ** 2
        return max(variance, 0.0) ** 0.5
