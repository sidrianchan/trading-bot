"""Multi-timeframe setup engine.

Pipeline (matches the spec):

1. **Daily** — classify trend bias (uptrend / downtrend / range) using
   EMA20, EMA50, SMA200. Range-bias names are skipped entirely.
2. **1H** — find a structural setup AT a key support/resistance level
   (pullback to support, consolidation breakout, flag continuation,
   reversal at resistance). Records the level and provides the candidate
   stop based on the structural low/high.
3. **15-min** — trigger fires when a candle pattern occurs at the 1H
   level OR when a volume-confirmed breakout closes through the 1H
   level. The 15-min bar's price becomes the limit-order entry.
4. **Score** — composite score (0-100) per ``signals/scoring.py``;
   threshold gate (default 65).
5. **Hold-type tag** — intraday vs swing decided by whether the structural
   level came from the Daily/1H stack (swing) or only from the 15-min
   trigger context (intraday).

Output: a list of ``TradeSetup`` records ranked by score, top-N kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd
from loguru import logger

from signals.indicators.trend import classify, TrendSnapshot
from signals.indicators.momentum import rsi as rsi_series, macd as macd_calc, macd_cross
from signals.indicators.volatility import (
    atr as atr_series,
    bollinger_bands,
    at_lower_band,
    at_upper_band,
)
from signals.indicators.volume import has_volume_confirmation, obv, obv_breaking_out
from signals.price_action.support_resistance import build_level_map, LevelMap, Level
from signals.price_action.candlesticks import detect_all
from signals.price_action.breakouts import consolidation_breakout, flag_breakout
from signals.scoring import score_setup, passes_threshold, ScoreCard
from data.earnings import is_within_blackout

Direction = Literal["long", "short"]
HoldType = Literal["intraday", "swing"]


@dataclass(frozen=True)
class TradeSetup:
    ticker: str
    direction: Direction
    hold_type: HoldType
    entry_price: float
    stop_price: float
    target1_price: float          # 1:1 R:R first profit leg
    target2_price: float          # next major S/R
    atr: float                    # 15-min ATR at entry (for trailing stop)
    score: float
    components: dict[str, float]
    sr_level: float
    pattern_detail: str = ""
    setup_date: date | None = None

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def reward_risk(self) -> float:
        return abs(self.target1_price - self.entry_price) / max(self.risk_per_share, 1e-9)


@dataclass
class _BarStack:
    """Multi-TF bar bundle for one ticker at one moment."""

    daily: pd.DataFrame
    hourly: pd.DataFrame
    bars_15m: pd.DataFrame


@dataclass
class SetupEngineConfig:
    # Trend / EMAs / ADX
    ema_fast: int = 20
    ema_slow: int = 50
    sma_long: int = 200
    adx_period: int = 14
    adx_min: float = 20.0
    swing_pivot_window: int = 5
    sr_cluster_pct: float = 0.005
    round_ladders: list[int] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100])

    # Indicators
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Breakouts
    consolidation_window: int = 10
    atr_short: int = 14
    atr_long: int = 50
    atr_contraction: float = 0.7
    breakout_volume_mult: float = 1.5

    # Scoring
    score_threshold: float = 65.0
    top_n_per_day: int = 5

    # Hard gates — these components must be ON for entry (in addition to
    # the score threshold). RSI / MACD remain weighted bonuses.
    require_at_sr: bool = True
    require_candle: bool = True
    require_volume: bool = True

    # Stops / hold
    max_stop_atr_mult: float = 1.5
    candle_at_sr_tolerance: float = 0.005
    earnings_blackout_days: int = 5


class SetupEngine:
    """Builds ``TradeSetup`` records by stacking Daily → 1H → 15-min context.

    Stateless across calls — feed it a snapshot of bars per ticker and it
    returns the setups present at that moment. The backtester drives this
    engine bar-by-bar; the live agent will call it once per 15-min interval.
    """

    def __init__(self, config: SetupEngineConfig | None = None):
        self.cfg = config or SetupEngineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        bars_per_ticker: dict[str, _BarStack | dict[str, pd.DataFrame]],
        *,
        as_of: date | None = None,
        skip_earnings: bool = True,
    ) -> list[TradeSetup]:
        """Score every ticker and return the top-N setups (above threshold)."""
        setups: list[TradeSetup] = []
        for ticker, stack in bars_per_ticker.items():
            stack = self._coerce(stack)
            if stack is None:
                continue
            try:
                ts = self._evaluate_one(ticker, stack, as_of=as_of, skip_earnings=skip_earnings)
            except Exception as exc:
                logger.debug(f"setup engine error on {ticker}: {exc}")
                continue
            if ts is not None:
                setups.append(ts)

        setups.sort(key=lambda s: s.score, reverse=True)
        return setups[: self.cfg.top_n_per_day]

    # ------------------------------------------------------------------
    # Per-ticker evaluation
    # ------------------------------------------------------------------

    def _evaluate_one(
        self,
        ticker: str,
        stack: _BarStack,
        *,
        as_of: date | None,
        skip_earnings: bool,
    ) -> TradeSetup | None:
        cfg = self.cfg

        # 1. Daily trend bias --------------------------------------------------
        trend = classify(
            stack.daily,
            ema_fast=cfg.ema_fast,
            ema_slow=cfg.ema_slow,
            sma_long=cfg.sma_long,
            adx_period=cfg.adx_period,
            adx_min=cfg.adx_min,
        )
        if trend is None or trend.bias == "range":
            return None
        direction: Direction = "long" if trend.bias == "uptrend" else "short"

        # 2. 1H setup search ---------------------------------------------------
        hourly = stack.hourly
        bars_15m = stack.bars_15m
        if len(hourly) < cfg.atr_long + cfg.consolidation_window:
            return None
        if len(bars_15m) < max(cfg.bb_period, cfg.atr_period) + 5:
            return None

        # Build the level map from 1H structure plus daily reference levels
        level_map = build_level_map(
            hourly,
            swing_window=cfg.swing_pivot_window,
            round_ladders=cfg.round_ladders,
            cluster_tolerance_pct=cfg.sr_cluster_pct,
            daily_for_reference=stack.daily,
        )
        if not level_map.levels:
            return None

        last_15m_close = float(bars_15m["close"].iloc[-1])
        nearby = level_map.near(last_15m_close, tolerance_pct=cfg.candle_at_sr_tolerance)

        # Aligned levels = supports for longs, resistances for shorts
        wanted_role = "support" if direction == "long" else "resistance"
        aligned_levels = [lvl for lvl in nearby if lvl.role == wanted_role]
        at_sr_level = bool(aligned_levels)
        # Prefer a daily-reference level (PDH/PDL/PWH/PWL) when one is present
        # — those promote the trade to a swing setup. Otherwise pick the
        # closest level for the structural stop placement.
        DAILY_REF_KINDS = {"pdh", "pdl", "pwh", "pwl"}
        daily_aligned = [lvl for lvl in aligned_levels if lvl.kind in DAILY_REF_KINDS]
        sr_level_obj = (daily_aligned[0] if daily_aligned
                        else aligned_levels[0] if aligned_levels else None)
        sr_price = sr_level_obj.price if sr_level_obj is not None else last_15m_close

        # 3. 15-min triggers ---------------------------------------------------
        candles = detect_all(bars_15m)
        bullish_candle = any(p.direction == "bullish" for p in candles)
        bearish_candle = any(p.direction == "bearish" for p in candles)
        candle_aligned = (direction == "long" and bullish_candle) or (
            direction == "short" and bearish_candle
        )
        candle_detail = ", ".join(p.name for p in candles) if candles else ""

        # Or volume-confirmed breakout on 1H window
        cons = consolidation_breakout(
            hourly,
            consolidation_window=cfg.consolidation_window,
            atr_short=cfg.atr_short,
            atr_long=cfg.atr_long,
            atr_contraction=cfg.atr_contraction,
            min_volume_multiple=cfg.breakout_volume_mult,
        )
        flag = flag_breakout(hourly, min_volume_multiple=cfg.breakout_volume_mult)
        breakout_aligned = (
            (cons.triggered and cons.direction == direction)
            or (flag.triggered and flag.direction == direction)
        )

        if not (candle_aligned or breakout_aligned):
            return None

        # 4. Indicator confirmations ------------------------------------------
        close_15m = bars_15m["close"]
        rsi_15 = rsi_series(close_15m, period=cfg.rsi_period)
        rsi_now = float(rsi_15.iloc[-1]) if not rsi_15.dropna().empty else 50.0
        # Strict: long entries require true oversold (RSI < rsi_oversold),
        # short entries require true overbought (RSI > rsi_overbought).
        if direction == "long":
            rsi_aligned = rsi_now < cfg.rsi_oversold
        else:
            rsi_aligned = rsi_now > cfg.rsi_overbought

        macd_vals = macd_calc(
            close_15m,
            fast=cfg.macd_fast,
            slow=cfg.macd_slow,
            signal_period=cfg.macd_signal,
        )
        cross = macd_cross(macd_vals)
        macd_aligned = (direction == "long" and cross == "bull") or (
            direction == "short" and cross == "bear"
        )
        # Treat "histogram already in direction" as confirmation when no fresh cross
        if not macd_aligned and not macd_vals.histogram.dropna().empty:
            h_last = float(macd_vals.histogram.iloc[-1])
            macd_aligned = (direction == "long" and h_last > 0) or (
                direction == "short" and h_last < 0
            )

        volume_confirmed = has_volume_confirmation(bars_15m, period=20, multiple=1.5)
        if not volume_confirmed and breakout_aligned:
            # Already had volume confirmation in the breakout detector
            volume_confirmed = True

        # 5a. Hard gates — components that MUST be on regardless of score.
        # Trend is already gated (we returned early if bias was 'range').
        if cfg.require_at_sr and not at_sr_level:
            return None
        if cfg.require_candle and not candle_aligned:
            return None
        if cfg.require_volume and not volume_confirmed:
            return None

        # 5b. Score (RSI / MACD remain weighted bonuses) ----------------------
        card = score_setup(
            at_sr_level=at_sr_level,
            candle_triggered=candle_aligned,
            trend_aligned=True,           # tautology: we filtered earlier
            rsi_aligned=rsi_aligned,
            macd_aligned=macd_aligned,
            volume_confirmed=volume_confirmed,
        )
        if not passes_threshold(card, threshold=cfg.score_threshold):
            return None

        # 6. Stop / target structure ------------------------------------------
        atr_15 = atr_series(bars_15m, period=cfg.atr_period)
        if atr_15.dropna().empty:
            return None
        atr_now = float(atr_15.iloc[-1])
        if atr_now <= 0:
            return None

        entry, stop, t1, t2 = self._build_stops_targets(
            direction=direction,
            entry_price=last_15m_close,
            sr_price=sr_price,
            atr_15=atr_now,
            level_map=level_map,
        )
        if stop is None or t1 is None:
            return None

        risk = abs(entry - stop)
        if risk < 0.01 or risk > cfg.max_stop_atr_mult * atr_now:
            return None
        if abs(t1 - entry) / max(risk, 1e-9) < 1.0:
            return None

        # 7. Hold-type classification -----------------------------------------
        # Swing if the structural level is a Daily reference (PDH/PDL/PWH/PWL).
        # Intraday otherwise (1H swing pivot or round-number level).
        hold_type: HoldType = (
            "swing"
            if (sr_level_obj is not None and sr_level_obj.kind in DAILY_REF_KINDS)
            else "intraday"
        )

        # Earnings filter (swing only) ----------------------------------------
        if hold_type == "swing" and skip_earnings and as_of is not None:
            try:
                if is_within_blackout(ticker, as_of, cfg.earnings_blackout_days):
                    return None
            except Exception as exc:
                logger.debug(f"earnings filter skipped for {ticker}: {exc}")

        return TradeSetup(
            ticker=ticker,
            direction=direction,
            hold_type=hold_type,
            entry_price=round(entry, 2),
            stop_price=round(stop, 2),
            target1_price=round(t1, 2),
            target2_price=round(t2, 2),
            atr=atr_now,
            score=card.total,
            components=card.components,
            sr_level=sr_price,
            pattern_detail=candle_detail or cons.detail or flag.detail,
            setup_date=as_of,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_stops_targets(
        self,
        *,
        direction: Direction,
        entry_price: float,
        sr_price: float,
        atr_15: float,
        level_map: LevelMap,
    ) -> tuple[float, float | None, float | None, float | None]:
        buffer = 0.1 * atr_15
        if direction == "long":
            stop = sr_price - buffer
            if stop >= entry_price:
                return entry_price, None, None, None
            t1 = entry_price + (entry_price - stop)
            # T2 = next resistance above entry
            resistances = sorted(
                (lvl.price for lvl in level_map.levels if lvl.price > entry_price)
            )
            t2 = resistances[0] if resistances else (entry_price + 2.0 * (entry_price - stop))
            return entry_price, stop, t1, t2
        # short
        stop = sr_price + buffer
        if stop <= entry_price:
            return entry_price, None, None, None
        t1 = entry_price - (stop - entry_price)
        supports = sorted(
            (lvl.price for lvl in level_map.levels if lvl.price < entry_price),
            reverse=True,
        )
        t2 = supports[0] if supports else (entry_price - 2.0 * (stop - entry_price))
        return entry_price, stop, t1, t2

    @staticmethod
    def _coerce(stack) -> _BarStack | None:
        if isinstance(stack, _BarStack):
            return stack
        if isinstance(stack, dict):
            try:
                return _BarStack(
                    daily=stack["daily"],
                    hourly=stack["1h"],
                    bars_15m=stack["15m"],
                )
            except KeyError:
                return None
        return None
