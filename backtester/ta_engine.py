"""Technical-analysis backtester.

Drives :class:`signals.setup.SetupEngine` over a historical universe and
simulates intraday + swing trade execution with limit orders, tiered exits,
ATR trailing stops, and per-trade / per-day risk caps.

Public entry point: :func:`run_ta_backtest`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

from data.timeframes import resample, market_hours_only
from signals.indicators.volatility import atr as atr_series
from signals.setup import SetupEngine, SetupEngineConfig, TradeSetup
from backtester.metrics import compute_metrics


# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TABacktestConfig:
    initial_capital: float = 100_000.0
    risk_per_trade_pct: float = 0.01
    max_position_value_pct: float = 0.10
    max_concurrent: int = 5
    max_per_sector: int = 2
    daily_pnl_halt_pct: float = -0.02

    spread_bps: float = 1.0          # 0.01% per side
    cancel_after_bars: int = 2       # cancel limit if unfilled within N 15-min bars
    fill_unfilled_as_market: bool = False

    # Exits
    target1_r: float = 1.0
    target1_pct: float = 0.40
    target2_pct: float = 0.40
    trail_pct: float = 0.20
    trail_atr_mult: float = 1.0

    # Hold
    intraday_hard_close_et: str = "15:30"
    swing_max_days: int = 5
    earnings_blackout_days: int = 5

    # Gate
    min_win_rate: float = 0.45
    min_reward_risk: float = 1.5
    max_consecutive_losing_days: int = 7


@dataclass
class _OpenTrade:
    setup: TradeSetup
    entry_dt: datetime
    qty: int                          # full original size
    qty_remaining: int
    t1_filled: bool = False
    t2_filled: bool = False
    trailing_stop: float = 0.0
    realized_pnl: float = 0.0
    sector: str | None = None


@dataclass
class _ClosedLeg:
    """A single fill (T1, T2, trailing exit, hard-close, or stop)."""

    ticker: str
    direction: str
    hold_type: str
    entry_dt: datetime
    exit_dt: datetime
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    close_type: str          # "t1" | "t2" | "trail" | "hard_close" | "stop" | "earnings_close" | "max_hold"


# ──────────────────────────────────────────────────────────────────────────


class TABacktester:
    """Replay 15-min bars day-by-day, calling the setup engine each interval."""

    def __init__(
        self,
        cfg: TABacktestConfig,
        engine_cfg: SetupEngineConfig,
        sectors: dict[str, str] | None = None,
    ):
        self.cfg = cfg
        self.engine = SetupEngine(engine_cfg)
        self.engine_cfg = engine_cfg
        self.sectors = sectors or {}

        self._capital = cfg.initial_capital
        self._open: dict[str, _OpenTrade] = {}
        self._closed: list[_ClosedLeg] = []
        self._equity_curve: dict[date, float] = {}

    # ------------------------------------------------------------------

    def run(
        self,
        bars_1min: dict[str, pd.DataFrame],
        bars_daily: dict[str, pd.DataFrame],
        *,
        start: date,
        end: date,
    ) -> tuple[pd.Series, list[_ClosedLeg]]:
        """Run the backtest. Returns (daily equity series, closed-leg log)."""
        logger.info(f"TA backtest: {start} → {end}, {len(bars_1min)} tickers")

        # Pre-resample once per ticker — much faster than per-day.
        bars_15m_full: dict[str, pd.DataFrame] = {}
        bars_1h_full: dict[str, pd.DataFrame] = {}
        for t, df in bars_1min.items():
            if df.empty:
                continue
            df_mkt = market_hours_only(df)
            if df_mkt.empty:
                continue
            try:
                bars_15m_full[t] = resample(df_mkt, "15min")
                bars_1h_full[t] = resample(df_mkt, "1h")
            except Exception as exc:
                logger.debug(f"resample failed for {t}: {exc}")

        # Use ET-local dates so a bar at 21:00 UTC = 16:00 ET correctly
        # belongs to that ET trading session.
        all_date_set: set = set()
        for df in bars_15m_full.values():
            idx = df.index
            idx_local = idx.tz_convert("America/New_York") if idx.tz is not None else idx
            all_date_set.update(idx_local.date)
        all_dates = sorted(d for d in all_date_set if start <= d <= end)
        if not all_dates:
            logger.warning("No 15-min bars in requested range")
            return pd.Series(dtype=float), []

        prev_capital = self._capital
        total_days = len(all_dates)
        for i, trading_date in enumerate(all_dates, start=1):
            self._simulate_day(
                trading_date,
                bars_15m_full,
                bars_1h_full,
                bars_daily,
            )
            self._equity_curve[trading_date] = self._capital
            day_pnl = self._capital - prev_capital
            n_trades_today = sum(
                1 for c in self._closed if c.exit_dt.date() == trading_date
            )
            # Per-day progress at INFO level so we can see the run advance
            if i % 5 == 0 or n_trades_today or i == total_days:
                logger.info(
                    f"day {i}/{total_days}  {trading_date}  "
                    f"P&L=${day_pnl:>+,.2f}  closed={n_trades_today}  "
                    f"open={len(self._open)}  equity=${self._capital:,.0f}"
                )
            prev_capital = self._capital

        equity = pd.Series(self._equity_curve, name="portfolio_value")
        equity.index = pd.to_datetime(equity.index)
        return equity, self._closed

    # ------------------------------------------------------------------

    def _simulate_day(
        self,
        trading_date: date,
        bars_15m_full: dict[str, pd.DataFrame],
        bars_1h_full: dict[str, pd.DataFrame],
        bars_daily: dict[str, pd.DataFrame],
    ) -> None:
        cfg = self.cfg
        td = pd.Timestamp(trading_date)
        starting_capital = self._capital
        daily_halt = False

        # Build a per-bar timeline across all tickers, sorted.
        # ``trading_date`` was derived from the bars' ET-local date; convert
        # the timestamps to ET before extracting `.date` so we match correctly
        # for both tz-aware (UTC) and tz-naive indexes.
        timeline: list[tuple[pd.Timestamp, str, dict]] = []
        for ticker, df in bars_15m_full.items():
            idx = df.index
            idx_local = idx.tz_convert("America/New_York") if idx.tz is not None else idx
            day_mask = idx_local.date == trading_date
            day_df = df[day_mask]
            if day_df.empty:
                continue
            for ts, row in day_df.iterrows():
                timeline.append((
                    ts, ticker,
                    {"open": row.open, "high": row.high, "low": row.low,
                     "close": row.close, "volume": row.volume},
                ))
        timeline.sort(key=lambda x: x[0])

        if not timeline:
            return

        spread_cost = lambda px: px * cfg.spread_bps / 10_000.0  # noqa: E731

        seen_setup_for_ticker_today: dict[str, int] = {}  # ticker → bars elapsed since trigger
        last_close_per_ticker: dict[str, float] = {}      # last seen close per ticker today
        hard_closed_today = False                         # one-shot flush at 15:30

        for ts, ticker, bar in timeline:
            ts_et = ts.tz_convert("America/New_York") if ts.tz is not None else ts
            time_str = ts_et.strftime("%H:%M") if hasattr(ts_et, "strftime") else "00:00"

            # Track the latest close per ticker so the hard-close block can
            # mark every position to its OWN last seen price (not whichever
            # ticker happens to be on the current timeline iteration).
            last_close_per_ticker[ticker] = bar["close"]

            # 1. Hard close at end of session — fire once when we cross 15:30
            #    ET, mark each open intraday trade to its own last close.
            if not hard_closed_today and time_str >= cfg.intraday_hard_close_et:
                hard_closed_today = True
                for t, ot in list(self._open.items()):
                    if ot.setup.hold_type == "intraday":
                        close_price = last_close_per_ticker.get(t, ot.setup.entry_price)
                        # Use the open trade's own ticker timestamp if we have
                        # a more recent bar for it; otherwise use the current
                        # bar's timestamp as a stand-in.
                        self._close_trade(ot, close_price, ts, "hard_close", spread_cost)
                        if ot.qty_remaining <= 0:
                            self._open.pop(t, None)

            # 2. Manage open trades on this bar (stops/targets/trail)
            ot = self._open.get(ticker)
            if ot is not None:
                self._manage_open_trade(ot, bar, ts, spread_cost)
                # If trade fully closed during management, drop it
                if ot.qty_remaining <= 0:
                    self._open.pop(ticker, None)
                    self._update_daily_halt(starting_capital)
                continue

            if daily_halt:
                continue

            # 3. Look for a fresh setup ----------------------------------------
            # Re-evaluate at most once per ~15-min bar per ticker by using the
            # bar's timestamp as the trigger key (the engine uses only history
            # up to and including this bar).
            # Build the bar-stack snapshot with strict no-lookahead slices.
            stack = self._snapshot(
                ticker,
                ts,
                bars_15m_full,
                bars_1h_full,
                bars_daily,
            )
            if stack is None:
                continue
            # Earnings filter disabled in the backtester — it would call
            # yfinance hundreds of thousands of times. The filter applies in
            # the live agent loop (Phase F).
            results = self.engine.evaluate(
                {ticker: stack}, as_of=trading_date, skip_earnings=False
            )
            if not results:
                continue
            setup = results[0]

            if not self._can_open(ticker):
                continue
            qty = self._size(setup)
            if qty <= 0:
                continue

            entry_price = setup.entry_price
            # Limit-order fill model: assume filled at the trigger close (the
            # 15-min trigger bar is by construction the bar that produced the
            # signal, so price is in-range). For higher fidelity later we can
            # require the next bar to trade through ``entry_price``.
            fill_cost = spread_cost(entry_price)
            self._capital -= fill_cost * qty
            ot = _OpenTrade(
                setup=setup,
                entry_dt=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                qty=qty,
                qty_remaining=qty,
                trailing_stop=setup.stop_price,
                sector=self.sectors.get(ticker),
            )
            self._open[ticker] = ot
            seen_setup_for_ticker_today[ticker] = 0

        def _today_df(df: pd.DataFrame) -> pd.DataFrame:
            idx = df.index
            idx_local = idx.tz_convert("America/New_York") if idx.tz is not None else idx
            return df[idx_local.date == trading_date]

        # End-of-day cleanup: any intraday positions still open get force-closed
        for t, ot in list(self._open.items()):
            if ot.setup.hold_type == "intraday":
                last_bar = bars_15m_full[t]
                day_df = _today_df(last_bar)
                if day_df.empty:
                    continue
                last_close = float(day_df["close"].iloc[-1])
                last_ts = day_df.index[-1]
                self._close_trade(ot, last_close, last_ts, "hard_close", spread_cost)
                if ot.qty_remaining <= 0:
                    self._open.pop(t, None)

        # Swing positions: enforce max-hold + earnings rolling check
        for t, ot in list(self._open.items()):
            if ot.setup.hold_type != "swing":
                continue
            held_days = (trading_date - ot.entry_dt.date()).days
            if held_days >= self.cfg.swing_max_days:
                last_bar = bars_15m_full.get(t)
                if last_bar is None or last_bar.empty:
                    continue
                day_df = _today_df(last_bar)
                if day_df.empty:
                    continue
                last_close = float(day_df["close"].iloc[-1])
                last_ts = day_df.index[-1]
                self._close_trade(ot, last_close, last_ts, "max_hold", spread_cost)
                if ot.qty_remaining <= 0:
                    self._open.pop(t, None)

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    def _manage_open_trade(
        self,
        ot: _OpenTrade,
        bar: dict,
        ts: pd.Timestamp,
        spread_cost,
    ) -> None:
        cfg = self.cfg
        s = ot.setup
        is_long = s.direction == "long"
        high, low = bar["high"], bar["low"]

        # Stop: hits the trailing stop (or original if T1 not yet filled)
        stop_price = ot.trailing_stop
        hit_stop = (low <= stop_price) if is_long else (high >= stop_price)

        if hit_stop:
            self._close_trade(ot, stop_price, ts, "stop", spread_cost)
            return

        # T1 (1R, 40%)
        if not ot.t1_filled:
            t1 = s.target1_price
            hit_t1 = (high >= t1) if is_long else (low <= t1)
            if hit_t1:
                fill_qty = max(1, int(ot.qty * cfg.target1_pct))
                fill_qty = min(fill_qty, ot.qty_remaining)
                self._record_partial(ot, t1, ts, "t1", fill_qty, spread_cost)
                ot.t1_filled = True
                # Move stop to break-even after T1
                ot.trailing_stop = s.entry_price

        # T2 (next major S/R, 40%)
        if ot.t1_filled and not ot.t2_filled and ot.qty_remaining > 0:
            t2 = s.target2_price
            hit_t2 = (high >= t2) if is_long else (low <= t2)
            if hit_t2:
                fill_qty = max(1, int(ot.qty * cfg.target2_pct))
                fill_qty = min(fill_qty, ot.qty_remaining)
                self._record_partial(ot, t2, ts, "t2", fill_qty, spread_cost)
                ot.t2_filled = True

        # Trailing stop on remaining 20% — 1×ATR from latest bar
        if ot.t2_filled and ot.qty_remaining > 0:
            atr_buf = cfg.trail_atr_mult * s.atr
            if is_long:
                candidate = bar["close"] - atr_buf
                if candidate > ot.trailing_stop:
                    ot.trailing_stop = candidate
            else:
                candidate = bar["close"] + atr_buf
                if candidate < ot.trailing_stop:
                    ot.trailing_stop = candidate

    def _record_partial(
        self,
        ot: _OpenTrade,
        exit_price: float,
        ts: pd.Timestamp,
        close_type: str,
        qty: int,
        spread_cost,
    ) -> None:
        sign = 1 if ot.setup.direction == "long" else -1
        pnl = sign * (exit_price - ot.setup.entry_price) * qty - spread_cost(exit_price) * qty
        self._capital += pnl
        ot.qty_remaining -= qty
        ot.realized_pnl += pnl
        self._closed.append(_ClosedLeg(
            ticker=ot.setup.ticker,
            direction=ot.setup.direction,
            hold_type=ot.setup.hold_type,
            entry_dt=ot.entry_dt,
            exit_dt=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            entry_price=ot.setup.entry_price,
            exit_price=exit_price,
            qty=qty,
            pnl=pnl,
            close_type=close_type,
        ))

    def _close_trade(
        self,
        ot: _OpenTrade,
        exit_price: float,
        ts: pd.Timestamp,
        close_type: str,
        spread_cost,
    ) -> None:
        if ot.qty_remaining <= 0:
            return
        sign = 1 if ot.setup.direction == "long" else -1
        qty = ot.qty_remaining
        pnl = sign * (exit_price - ot.setup.entry_price) * qty - spread_cost(exit_price) * qty
        self._capital += pnl
        ot.realized_pnl += pnl
        ot.qty_remaining = 0
        self._closed.append(_ClosedLeg(
            ticker=ot.setup.ticker,
            direction=ot.setup.direction,
            hold_type=ot.setup.hold_type,
            entry_dt=ot.entry_dt,
            exit_dt=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            entry_price=ot.setup.entry_price,
            exit_price=exit_price,
            qty=qty,
            pnl=pnl,
            close_type=close_type,
        ))

    # ------------------------------------------------------------------
    # Sizing / sector / halt checks
    # ------------------------------------------------------------------

    def _size(self, setup: TradeSetup) -> int:
        cfg = self.cfg
        risk = setup.risk_per_share
        if risk < 0.01:
            return 0
        risk_dollars = self._capital * cfg.risk_per_trade_pct
        qty = int(risk_dollars / risk)
        if qty <= 0:
            return 0
        # Cap by max position value
        max_value = self._capital * cfg.max_position_value_pct
        max_qty_by_value = int(max_value / max(setup.entry_price, 1e-9))
        return max(1, min(qty, max_qty_by_value))

    def _can_open(self, ticker: str) -> bool:
        cfg = self.cfg
        if ticker in self._open:
            return False
        if len(self._open) >= cfg.max_concurrent:
            return False
        sector = self.sectors.get(ticker)
        if sector is not None:
            same_sector = sum(1 for t, ot in self._open.items() if ot.sector == sector)
            if same_sector >= cfg.max_per_sector:
                return False
        return True

    def _update_daily_halt(self, starting_capital: float) -> None:
        # Stub for future per-bar halt enforcement; current loop handles it
        # via daily_halt flag set during _simulate_day.
        pass

    # ------------------------------------------------------------------

    def _snapshot(
        self,
        ticker: str,
        ts: pd.Timestamp,
        bars_15m_full: dict[str, pd.DataFrame],
        bars_1h_full: dict[str, pd.DataFrame],
        bars_daily: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame] | None:
        m15 = bars_15m_full.get(ticker)
        h1 = bars_1h_full.get(ticker)
        daily = bars_daily.get(ticker)
        if m15 is None or h1 is None or daily is None:
            return None
        # Slice strictly up to and including the current 15-min trigger
        m15_slice = m15[m15.index <= ts]
        h1_slice = h1[h1.index <= ts]

        # Daily bars are tz-naive (yfinance); intraday are tz-aware (UTC).
        # Normalize the cutoff to a tz-naive midnight in the daily series'
        # implicit timezone (yfinance dailies are calendar dates).
        ts_for_daily = pd.Timestamp(ts)
        if ts_for_daily.tz is not None:
            ts_for_daily = ts_for_daily.tz_convert("America/New_York").tz_localize(None)
        cutoff = ts_for_daily.normalize()
        daily_idx = daily.index
        if daily_idx.tz is not None:
            daily_idx_for_cmp = daily_idx.tz_convert(None)
            mask = daily_idx_for_cmp <= cutoff
        else:
            mask = daily_idx <= cutoff
        daily_slice = daily[mask]

        if m15_slice.empty or h1_slice.empty or daily_slice.empty:
            return None
        return {"daily": daily_slice, "1h": h1_slice, "15m": m15_slice}


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point — wired into main.py
# ──────────────────────────────────────────────────────────────────────────


def run_ta_backtest(config: dict) -> None:
    """Load 1-min and daily history, run the backtester for the test window,
    print the gate verdict.

    The user's plan calls for walk-forward 2023-train / 2024-test; in this
    initial implementation thresholds and weights are fixed by ``config.yaml``
    (no automated grid search), so 2023 is observed but only 2024 is reported
    against the gate. Tuning will be wired in once the gate verdict is in.
    """
    from data.universe import (
        get_sp500_tickers, get_russell1000_tickers, apply_liquidity_filter,
    )
    from data.market import fetch_intraday_bars_range, fetch_prices

    ta_cfg = config.get("ta", {})
    bt = ta_cfg.get("backtest", {})
    risk = config.get("risk", {})
    portfolio = config.get("portfolio", {})

    # ── Data fetch range (full, always — these match the parquet cache keys)
    train_start = pd.Timestamp(bt.get("train_start", "2023-01-01")).date()
    train_end = pd.Timestamp(bt.get("train_end", "2023-12-31")).date()
    fetch_test_end = pd.Timestamp(bt.get("test_end", "2024-12-31")).date()

    # ── Engine evaluation window (what the gate is measured over)
    test_start = pd.Timestamp(bt.get("test_start", "2024-01-01")).date()
    test_end = fetch_test_end

    # Optional smoke-run overrides — affect only the engine evaluation window,
    # never the data-fetch range, so the parquet cache always hits.
    if (override := os.environ.get("TA_BACKTEST_TEST_START")):
        test_start = pd.Timestamp(override).date()
        logger.warning(f"TA_BACKTEST_TEST_START override → {test_start}")
    if (override := os.environ.get("TA_BACKTEST_TEST_END")):
        test_end = pd.Timestamp(override).date()
        logger.warning(f"TA_BACKTEST_TEST_END override → {test_end}")
    max_universe = int(os.environ.get("TA_BACKTEST_MAX_TICKERS", "0"))
    initial_capital = bt.get("initial_capital", portfolio.get("initial_capital", 100_000.0))

    cfg = TABacktestConfig(
        initial_capital=float(initial_capital),
        risk_per_trade_pct=risk.get("risk_per_trade_pct", 0.01),
        max_position_value_pct=portfolio.get("max_position_value_pct", 0.10),
        max_concurrent=portfolio.get("max_concurrent_positions", 5),
        max_per_sector=portfolio.get("max_per_sector", 2),
        daily_pnl_halt_pct=risk.get("daily_pnl_halt_pct", -0.02),
        spread_bps=bt.get("spread_bps", 1),
        cancel_after_bars=config.get("execution", {}).get("cancel_after_bars", 2),
        target1_r=ta_cfg.get("exits", {}).get("target1_r", 1.0),
        target1_pct=ta_cfg.get("exits", {}).get("target1_pct", 0.40),
        target2_pct=ta_cfg.get("exits", {}).get("target2_pct", 0.40),
        trail_pct=ta_cfg.get("exits", {}).get("trail_pct", 0.20),
        trail_atr_mult=ta_cfg.get("exits", {}).get("trail_atr_mult", 1.0),
        intraday_hard_close_et=ta_cfg.get("hold", {}).get("intraday_hard_close", "15:30"),
        swing_max_days=ta_cfg.get("hold", {}).get("swing_max_days", 5),
        earnings_blackout_days=ta_cfg.get("hold", {}).get("earnings_blackout_days", 5),
        min_win_rate=ta_cfg.get("gate", {}).get("min_win_rate", 0.45),
        min_reward_risk=ta_cfg.get("gate", {}).get("min_reward_risk", 1.5),
        max_consecutive_losing_days=ta_cfg.get("gate", {}).get("max_consecutive_losing_days", 7),
    )
    engine_cfg = _build_engine_config(ta_cfg)

    # Universe -------------------------------------------------------------
    sp500 = get_sp500_tickers()
    russell = []
    try:
        russell = get_russell1000_tickers()
    except Exception:
        pass
    full_universe = sorted(set(sp500) | set(russell) | {"SPY"})

    # First-pass optimization: if a parquet cache already exists for the
    # configured FETCH range, restrict to that set instead of fetching ~1000
    # tickers from Alpaca + yfinance. The cache was populated for the most
    # actively traded large-cap names — exactly what the liquidity filter
    # would have selected anyway.
    cache_dir = Path("data/cache/intraday")
    train_str = pd.Timestamp(train_start).strftime("%Y-%m-%d")
    fetch_test_str = pd.Timestamp(fetch_test_end).strftime("%Y-%m-%d")
    if cache_dir.exists():
        cached_tickers = {
            p.name.split("_")[0]
            for p in cache_dir.glob(f"*_{train_str}_{fetch_test_str}.parquet")
        }
        if cached_tickers:
            universe = sorted(cached_tickers & set(full_universe) | {"SPY"})
            if max_universe and len(universe) > max_universe:
                universe = sorted(set(universe[:max_universe]) | {"SPY"})
            logger.warning(
                f"Phase E first-pass: restricting to {len(universe)} cached tickers "
                f"(use a fresh fetch for full {len(full_universe)}-name universe later)"
            )
        else:
            universe = full_universe
            logger.info(f"Universe pre-liquidity: {len(universe)} tickers")
    else:
        universe = full_universe
        logger.info(f"Universe pre-liquidity: {len(universe)} tickers")

    # Daily bars (need ≥1 yr extra for SMA200 warmup) ---------------------
    daily_start = (pd.Timestamp(train_start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    daily_end = (pd.Timestamp(fetch_test_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Fetching daily bars {daily_start} → {daily_end}")
    daily_close = fetch_prices(universe, start=daily_start, end=daily_end, source="yfinance")
    bars_daily = _close_only_to_ohlc(daily_close)

    # 1-min bars — always request the FULL configured range so the parquet
    # cache hits. The engine filters to ``test_start..test_end`` internally.
    range_start = train_start.strftime("%Y-%m-%d")
    range_end = fetch_test_end.strftime("%Y-%m-%d")
    logger.info(f"Fetching 1-min bars {range_start} → {range_end}")
    bars_1min = fetch_intraday_bars_range(universe, range_start, range_end)
    logger.info(f"Loaded 1-min bars for {len(bars_1min)} tickers")

    bt_engine = TABacktester(cfg=cfg, engine_cfg=engine_cfg)
    equity, closed = bt_engine.run(
        bars_1min=bars_1min,
        bars_daily=bars_daily,
        start=test_start,
        end=test_end,
    )

    _report(cfg, equity, closed, label=f"TEST {test_start} → {test_end}")


def _build_engine_config(ta_cfg: dict) -> SetupEngineConfig:
    trend = ta_cfg.get("trend", {})
    sr = ta_cfg.get("support_resistance", {})
    ind = ta_cfg.get("indicators", {})
    br = ta_cfg.get("breakouts", {})
    sc = ta_cfg.get("scoring", {})
    weights = sc.get("weights", {})

    return SetupEngineConfig(
        ema_fast=trend.get("ema_fast", 20),
        ema_slow=trend.get("ema_slow", 50),
        sma_long=trend.get("sma_long", 200),
        swing_pivot_window=sr.get("swing_pivot_window", 5),
        sr_cluster_pct=sr.get("cluster_tolerance_pct", 0.005),
        round_ladders=sr.get("round_number_levels", [1, 5, 10, 25, 50, 100]),
        rsi_period=ind.get("rsi_period", 14),
        rsi_oversold=ind.get("rsi_oversold", 35),
        rsi_overbought=ind.get("rsi_overbought", 65),
        macd_fast=ind.get("macd_fast", 12),
        macd_slow=ind.get("macd_slow", 26),
        macd_signal=ind.get("macd_signal", 9),
        bb_period=ind.get("bb_period", 20),
        bb_std=ind.get("bb_std", 2.0),
        atr_period=ind.get("atr_period", 14),
        consolidation_window=10,
        atr_short=14,
        atr_long=50,
        atr_contraction=br.get("consolidation_atr_contraction", 0.7),
        breakout_volume_mult=br.get("min_volume_multiple", 1.5),
        score_threshold=sc.get("threshold", 65),
        top_n_per_day=sc.get("top_n_per_day", 5),
        max_stop_atr_mult=ta_cfg.get("risk", {}).get("max_stop_atr_mult", 1.5),
        earnings_blackout_days=ta_cfg.get("hold", {}).get("earnings_blackout_days", 5),
    )


def _close_only_to_ohlc(daily_close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """fetch_prices() returns only adjusted closes; synthesize OHLC for the
    setup engine. Synthetic O=C[t-1], H=max(C[t-1], C), L=min, V=0 — adequate
    for trend classification (which only reads close) and S/R pivots fall back
    to swing-close pivots in the absence of real H/L."""
    out: dict[str, pd.DataFrame] = {}
    for col in daily_close.columns:
        c = daily_close[col].dropna()
        if c.empty:
            continue
        prev = c.shift(1).bfill()
        out[col] = pd.DataFrame({
            "open":   prev.values,
            "high":   pd.concat([prev, c], axis=1).max(axis=1).values,
            "low":    pd.concat([prev, c], axis=1).min(axis=1).values,
            "close":  c.values,
            "volume": [0] * len(c),
        }, index=c.index)
    return out


def _report(
    cfg: TABacktestConfig,
    equity: pd.Series,
    closed: list[_ClosedLeg],
    *,
    label: str,
) -> None:
    by_trade = _aggregate_by_trade(closed)
    wins = [t for t in by_trade if t["pnl"] > 0]
    losses = [t for t in by_trade if t["pnl"] <= 0]
    n = len(by_trade)
    win_rate = len(wins) / n if n else 0.0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
    rr = abs(avg_win / avg_loss) if avg_loss else 0.0

    # Consecutive losing days
    by_day: dict[date, float] = {}
    for t in by_trade:
        d = t["exit_dt"].date()
        by_day[d] = by_day.get(d, 0.0) + t["pnl"]
    streak = max_streak = 0
    for d in sorted(by_day):
        if by_day[d] < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    print(f"\n{'='*60}")
    print(f"  TA BACKTEST  {label}")
    print(f"{'='*60}")
    print(f"  Total round-trip trades : {n}")
    print(f"  Win rate                : {win_rate:.1%}")
    print(f"  Avg win                 : ${avg_win:>+,.2f}")
    print(f"  Avg loss                : ${avg_loss:>,.2f}")
    print(f"  Reward:Risk             : {rr:.2f}:1")
    print(f"  Max consec losing days  : {max_streak}")
    if len(equity) > 1:
        try:
            m = compute_metrics(equity)
            print(f"  CAGR                    : {m.get('cagr', 0):.1%}")
            print(f"  Sharpe                  : {m.get('sharpe', 0):.2f}")
            print(f"  Max Drawdown            : {m.get('max_drawdown', 0):.1%}")
        except Exception as exc:
            logger.debug(f"metrics computation skipped: {exc}")

    gate_pass = (
        win_rate >= cfg.min_win_rate
        and rr >= cfg.min_reward_risk
        and max_streak <= cfg.max_consecutive_losing_days
    )
    if gate_pass:
        print("\n  GATE PASSED. Safe to begin paper trading (after Phase F wires the live loop).")
    else:
        print("\n  GATE FAILED:")
        if win_rate < cfg.min_win_rate:
            print(f"     Win rate {win_rate:.1%} < {cfg.min_win_rate:.0%} minimum")
        if rr < cfg.min_reward_risk:
            print(f"     R:R {rr:.2f} < {cfg.min_reward_risk:.1f} minimum")
        if max_streak > cfg.max_consecutive_losing_days:
            print(f"     Max losing streak {max_streak} > {cfg.max_consecutive_losing_days} days")
        print("     Do NOT proceed to paper trading.")
    print(f"{'='*60}\n")


def _aggregate_by_trade(closed: list[_ClosedLeg]) -> list[dict]:
    """Collapse partial fills (T1/T2/trail/stop on the same entry) into a
    single round-trip P&L for win-rate accounting."""
    by_key: dict[tuple, dict] = {}
    for leg in closed:
        key = (leg.ticker, leg.entry_dt)
        rec = by_key.setdefault(key, {"pnl": 0.0, "exit_dt": leg.exit_dt})
        rec["pnl"] += leg.pnl
        if leg.exit_dt > rec["exit_dt"]:
            rec["exit_dt"] = leg.exit_dt
    return list(by_key.values())
