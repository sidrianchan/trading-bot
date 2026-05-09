from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from loguru import logger

from signals.intraday_composite import IntradayComposite
from signals.gap import StockSnapshot


@dataclass
class IntradayBacktestConfig:
    initial_capital: float = 100_000.0
    slippage_bps: float = 2.0        # per side (0.02%)
    spread_bps: float = 1.0          # one-way spread cost (0.01%)
    risk_per_trade_pct: float = 0.005
    max_concurrent: int = 8
    hard_close_time: str = "15:30"
    entry_start_time: str = "10:00"  # no entries until day type is classified
    day_classify_time: str = "10:00" # SPY move measured from 09:30 to this time
    orb_minutes: int = 15
    gap_min_pct: float = 1.5
    volume_ratio_min: float = 2.0
    orb_volume_confirm_ratio: float = 0.0
    atr_min_dollars: float = 0.50
    vwap_std_threshold: float = 1.5
    vwap_stop_std: float = 2.5                  # stop = VWAP ± vwap_stop_std × σ
    vwap_min_dollar_deviation_pct: float = 0.015  # 1.5% min |price - VWAP| / price
    vwap_confirm_reversal: bool = True          # require prior bar to turn back toward VWAP
    cooldown_bars_after_stop: int = 15          # block re-entry on same ticker for N bars after a stop
    max_trades_per_ticker_per_day: int = 2      # hard cap on entries per name per session
    day_trend_threshold: float = 0.01           # |SPY move| ≥ 1% by classify_time = trend
    spy_range_threshold: float = 0.005          # SPY high-low range < 0.5% reinforces "range"
    trend_day_size_mult: float = 0.5            # 50% size on trend days (0.0 = skip)
    vix_range_override: float = 20.0            # prior-day VIX > this → force "range"
    top_n_candidates: int = 15
    daily_pnl_halt_pct: float = -0.02
    min_rr_ratio: float = 1.5                   # safety: skip trades where structural R:R<1.5


@dataclass
class TradeResult:
    ticker: str
    date: date
    direction: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    close_type: str    # "target", "stop", "hard_close"
    hold_bars: int


class IntradayBacktester:
    """Event-driven 1-minute bar backtester for VWAP-mean-reversion strategy.

    Replays historical bars day by day. For each day:
      1. Builds snapshot-style data from the daily bar (prev close, ATR, gap)
      2. Runs the gap screen and candidate selection (universe shaping only)
      3. Replays 1-min bars; entries gated until `entry_start_time`
      4. At `day_classify_time`, classifies day from SPY open→that-time move
         (and optional prior-day VIX override) → "range" or "trend"
      5. VWAP mean-reversion entries: stop = VWAP ± vwap_stop_std·σ,
         target = VWAP itself (mean-reversion bracket, not fixed R:R)
      6. Trend day: position size × trend_day_size_mult (0.5 default)
      7. Bracket fills tracked per bar; hard close at `hard_close_time`
    """

    def __init__(self, config: IntradayBacktestConfig):
        self.cfg = config
        self._trade_log: list[TradeResult] = []
        self._vix_prev_close: dict[date, float] = {}

    # ------------------------------------------------------------------
    # Optional VIX prior-day-close enrichment
    # ------------------------------------------------------------------

    def _load_vix(self, start: date, end: date) -> None:
        """Fetch ^VIX daily closes via yfinance; map trading_date → prior close.

        Best-effort — silently no-ops on network/parsing failure (engine
        falls back to SPY-only day classification).
        """
        try:
            import yfinance as yf
            buf_start = pd.Timestamp(start) - pd.Timedelta(days=10)
            df = yf.download(
                "^VIX",
                start=buf_start.strftime("%Y-%m-%d"),
                end=(pd.Timestamp(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
            if df is None or df.empty:
                logger.warning("VIX fetch returned empty — day classifier will use SPY only")
                return
            close_col = df["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            shifted = close_col.shift(1).dropna()
            self._vix_prev_close = {ts.date(): float(v) for ts, v in shifted.items()}
            logger.info(f"VIX loaded: {len(self._vix_prev_close)} day(s) of prior closes")
        except Exception as e:
            logger.warning(f"VIX fetch failed ({e}) — day classifier will use SPY only")

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self, bars: dict[str, pd.DataFrame]) -> tuple[pd.Series, list[TradeResult]]:
        """Run the full backtest. Returns (daily_pnl_series, trade_log)."""
        self._trade_log = []
        capital = self.cfg.initial_capital

        all_dates = sorted({ts.date() for df in bars.values() for ts in df.index})
        if not all_dates:
            return pd.Series(dtype=float), []

        self._load_vix(all_dates[0], all_dates[-1])

        all_ts = pd.DatetimeIndex(all_dates)
        daily_values: dict[date, float] = {}
        daily_values[all_ts[0] - pd.Timedelta(days=1)] = capital   # seed

        for trading_date in all_ts:
            day_pnl, trades = self._simulate_day(bars, trading_date, capital)
            capital += day_pnl
            daily_values[trading_date] = capital
            self._trade_log.extend(trades)

            if trades:
                wins = sum(1 for t in trades if t.pnl > 0)
                logger.debug(
                    f"{trading_date}  P&L=${day_pnl:>+,.2f}  "
                    f"trades={len(trades)} ({wins}W/{len(trades)-wins}L)"
                )

        daily_pnl = pd.Series(daily_values, name="portfolio_value")
        daily_pnl.index = pd.to_datetime(daily_pnl.index)
        return daily_pnl, self._trade_log

    # ------------------------------------------------------------------
    # Per-day simulation
    # ------------------------------------------------------------------

    def _simulate_day(
        self,
        bars: dict[str, pd.DataFrame],
        trading_date,
        capital: float,
    ) -> tuple[float, list[TradeResult]]:
        cfg = self.cfg
        day_pnl = 0.0
        trades: list[TradeResult] = []
        open_trades: dict[str, _SimTrade] = {}
        daily_halt = False

        snapshots = self._build_snapshots(bars, trading_date)
        if not snapshots:
            return 0.0, []

        composite = IntradayComposite(
            top_n=cfg.top_n_candidates,
            gap_min_pct=cfg.gap_min_pct,
            volume_ratio_min=cfg.volume_ratio_min,
            atr_min_dollars=cfg.atr_min_dollars,
            orb_minutes=cfg.orb_minutes,
            vwap_std_threshold=cfg.vwap_std_threshold,
            vwap_min_dollar_deviation_pct=cfg.vwap_min_dollar_deviation_pct,
            vwap_confirm_reversal=cfg.vwap_confirm_reversal,
            orb_volume_confirm_ratio=cfg.orb_volume_confirm_ratio,
        )
        candidates = composite.rank_candidates(snapshots)
        if not candidates:
            return 0.0, []

        td = pd.Timestamp(trading_date)

        def _today_mkt(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                return df.iloc[0:0]
            idx = df.index
            day_idx = idx.floor("D")
            if day_idx.tz is not None:
                day_idx = day_idx.tz_convert(None)
            h, m = idx.hour, idx.minute
            after_open   = (h > 14) | ((h == 14) & (m >= 30))
            before_close = h < 21
            return df[(day_idx == td) & after_open & before_close]

        # Collect all 1-min market-hours bars for this day, sorted by timestamp
        day_bars: list[tuple[str, pd.Timestamp, dict]] = []
        for c in candidates:
            ticker = c.ticker
            if ticker not in bars:
                continue
            day_rows = _today_mkt(bars[ticker])
            for ts, row in day_rows.iterrows():
                day_bars.append((
                    ticker, ts,
                    {"open": row.open, "high": row.high, "low": row.low,
                     "close": row.close, "volume": row.volume}
                ))
        day_bars.sort(key=lambda x: x[1])

        # Pre-compute SPY day classification once (deterministic, lookahead-safe
        # because we only look at SPY bars up to and including classify_time)
        spy_today = _today_mkt(bars.get("SPY", pd.DataFrame()))
        precomputed_day_type = self._classify_day_from_spy(spy_today, trading_date, cfg)
        day_classified = False

        last_close: dict[str, float] = {}
        last_stop_bar: dict[str, int] = {}     # ticker → bar_count when last stopped out
        trades_today:  dict[str, int] = {}     # ticker → # entries opened today
        bar_count = 0
        for ticker, ts, bar in day_bars:
            bar_count += 1
            last_close[ticker] = bar["close"]

            ts_et = ts.tz_convert("America/New_York") if ts.tzinfo is not None else ts
            time_str = ts_et.strftime("%H:%M")

            # Hard close
            if time_str >= cfg.hard_close_time:
                for sim_trade in list(open_trades.values()):
                    close_price = last_close.get(sim_trade.ticker, sim_trade.entry_price)
                    result = sim_trade.close(close_price, "hard_close", cfg)
                    trades.append(result)
                    day_pnl += result.pnl
                open_trades.clear()
                break

            # Per-bar exits for open trades (always check, even before entry window)
            sim_trade = open_trades.get(ticker)
            if sim_trade:
                closed, result = sim_trade.check_exit(bar, cfg)
                if closed:
                    trades.append(result)
                    day_pnl += result.pnl
                    del open_trades[ticker]
                    if result.close_type == "stop":
                        last_stop_bar[ticker] = bar_count
                    if day_pnl / capital <= cfg.daily_pnl_halt_pct:
                        daily_halt = True
                continue

            # Always feed bars to build VWAP/ORB state
            signal = composite.on_bar(ticker, bar)

            # Day classification — fires once at classify_time
            if not day_classified and time_str >= cfg.day_classify_time:
                composite.set_day_type(precomputed_day_type)
                day_classified = True

            # No entries before classification window
            if not day_classified or time_str < cfg.entry_start_time:
                continue
            if daily_halt:
                continue
            if signal is None:
                continue
            if len(open_trades) >= cfg.max_concurrent:
                continue

            # Per-ticker daily cap and post-stop cooldown
            if trades_today.get(ticker, 0) >= cfg.max_trades_per_ticker_per_day:
                continue
            if bar_count - last_stop_bar.get(ticker, -10**9) < cfg.cooldown_bars_after_stop:
                continue

            cand = next((c for c in candidates if c.ticker == ticker), None)
            if cand is None:
                continue

            vwap = composite.get_vwap(ticker)
            std  = composite.get_vwap_std(ticker)
            if vwap is None or std is None or std <= 0:
                continue

            entry, stop, target, direction = self._calc_vwap_bracket(
                signal, bar["close"], vwap, std, cfg
            )
            if entry is None:
                continue

            risk_per_share = abs(entry - stop)
            reward_per_share = abs(target - entry)
            if risk_per_share < 0.01 or reward_per_share <= 0:
                continue
            rr = reward_per_share / risk_per_share
            if rr < cfg.min_rr_ratio:
                continue

            # Trend day → reduce or skip
            day_type = composite.get_day_type()
            size_mult = cfg.trend_day_size_mult if day_type == "trend" else 1.0
            if size_mult <= 0:
                continue

            base_qty = max(1, int(capital * cfg.risk_per_trade_pct / risk_per_share))
            qty = max(1, int(base_qty * size_mult))

            open_trades[ticker] = _SimTrade(
                ticker=ticker,
                date=trading_date,
                direction=direction,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                qty=qty,
                entry_bar=bar_count,
                atr=cand.atr,
            )
            trades_today[ticker] = trades_today.get(ticker, 0) + 1

        return day_pnl, trades

    @staticmethod
    def _mkt_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Filter to regular market hours: 09:30–16:00 ET = 14:30–21:00 UTC."""
        idx = df.index
        h, m = idx.hour, idx.minute
        after_open   = (h > 14) | ((h == 14) & (m >= 30))
        before_close = h < 21
        return df[after_open & before_close]

    def _build_snapshots(self, bars: dict[str, pd.DataFrame], trading_date) -> list[StockSnapshot]:
        """Build proxy snapshots from 1-min bar data for the morning scan."""
        snaps: list[StockSnapshot] = []
        for ticker, df in bars.items():
            day_idx = df.index.floor("D")
            if day_idx.tz is not None:
                day_idx = day_idx.tz_convert(None)
            td = pd.Timestamp(trading_date)

            df_mkt = self._mkt_hours(df)
            if df_mkt.empty:
                continue
            mkt_day_idx = df_mkt.index.floor("D")
            if mkt_day_idx.tz is not None:
                mkt_day_idx = mkt_day_idx.tz_convert(None)

            prev_rows  = df_mkt[mkt_day_idx < td]
            today_rows = df_mkt[mkt_day_idx == td]
            if prev_rows.empty or today_rows.empty:
                continue

            prev_close = float(prev_rows.iloc[-1]["close"])
            today_open = float(today_rows.iloc[0]["open"])
            gap_pct    = (today_open / prev_close) - 1.0 if prev_close > 0 else 0.0

            prev_day_idx = prev_rows.index.floor("D")
            if prev_day_idx.tz is not None:
                prev_day_idx = prev_day_idx.tz_convert(None)
            daily_hl = prev_rows.groupby(prev_day_idx).apply(
                lambda g: g["high"].max() - g["low"].min(), include_groups=False
            ).tail(14)
            atr = float(daily_hl.mean()) if len(daily_hl) >= 5 else 0.0

            daily_vol = prev_rows.groupby(prev_day_idx)["volume"].sum()
            prev_vol  = int(daily_vol.iloc[-1]) if not daily_vol.empty else 0

            open_bar_vol = float(today_rows.iloc[0]["volume"])
            avg_per_min  = float(prev_vol) / 390.0 if prev_vol > 0 else 1.0
            volume_ratio = open_bar_vol / avg_per_min if avg_per_min > 0 else 1.0

            snaps.append(StockSnapshot(
                ticker=ticker,
                prev_close=prev_close,
                latest_price=today_open,
                pre_market_volume=int(open_bar_vol),
                avg_volume_30d=float(prev_vol),
                atr=atr,
                gap_pct=gap_pct,
                volume_ratio=volume_ratio,
            ))
        return snaps

    def _classify_day_from_spy(self, spy_today: pd.DataFrame, trading_date, cfg) -> str:
        """Classify the day from SPY's open through `day_classify_time`.

        Returns "trend" if |SPY move from open| ≥ day_trend_threshold,
        "range" otherwise. Prior-day VIX > vix_range_override forces "range".
        """
        vix_prev = self._vix_prev_close.get(trading_date)
        if vix_prev is not None and vix_prev > cfg.vix_range_override:
            return "range"
        if spy_today.empty:
            return "range"

        # Bars at or before classify_time (UTC bars; convert to ET for cmp)
        idx = spy_today.index
        ts_et = idx.tz_convert("America/New_York") if idx.tz is not None else idx
        h = ts_et.hour
        m = ts_et.minute
        cls_h, cls_m = (int(x) for x in cfg.day_classify_time.split(":"))
        mask = (h < cls_h) | ((h == cls_h) & (m <= cls_m))
        window = spy_today[mask]
        if window.empty:
            return "range"

        spy_open = float(window.iloc[0]["open"])
        spy_now  = float(window.iloc[-1]["close"])
        if spy_open <= 0:
            return "range"
        move = abs(spy_now / spy_open - 1)
        return "trend" if move >= cfg.day_trend_threshold else "range"

    @staticmethod
    def _calc_vwap_bracket(signal, price, vwap, std, cfg):
        """VWAP mean-reversion bracket: target = VWAP, stop = VWAP ± stop_std·σ."""
        stop_band = cfg.vwap_stop_std * std

        if signal == "fade_long":
            # Price has dropped well below VWAP — fade back up toward VWAP
            if price >= vwap:
                return None, None, None, None
            entry  = price
            stop   = vwap - stop_band         # just beyond entry trigger band
            target = vwap                     # mean reversion to VWAP
            direction = "long"
            if stop >= entry or target <= entry:
                return None, None, None, None
            return entry, stop, target, direction

        if signal == "fade_short":
            if price <= vwap:
                return None, None, None, None
            entry  = price
            stop   = vwap + stop_band
            target = vwap
            direction = "short"
            if stop <= entry or target >= entry:
                return None, None, None, None
            return entry, stop, target, direction

        return None, None, None, None


@dataclass
class _SimTrade:
    ticker: str
    date: date
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    qty: int
    entry_bar: int
    atr: float = 0.0
    bar_count: int = 0

    def check_exit(self, bar: dict, cfg: IntradayBacktestConfig) -> tuple[bool, TradeResult | None]:
        self.bar_count += 1
        high = bar["high"]
        low  = bar["low"]

        hit_stop   = (low  <= self.stop_price)   if self.direction == "long" else (high >= self.stop_price)
        hit_target = (high >= self.target_price) if self.direction == "long" else (low  <= self.target_price)

        if hit_target:
            return True, self.close(self.target_price, "target", cfg)
        if hit_stop:
            return True, self.close(self.stop_price, "stop", cfg)
        return False, None

    def close(self, exit_price: float, close_type: str, cfg: IntradayBacktestConfig) -> TradeResult:
        slippage = exit_price * cfg.slippage_bps / 10_000
        spread   = exit_price * cfg.spread_bps / 10_000
        cost_per_share = slippage + spread

        direction_sign = 1 if self.direction == "long" else -1
        raw_pnl = direction_sign * (exit_price - self.entry_price) * self.qty
        costs   = cost_per_share * self.qty * 2   # round-trip
        net_pnl = raw_pnl - costs

        return TradeResult(
            ticker=self.ticker,
            date=self.date,
            direction=self.direction,
            entry_price=self.entry_price,
            exit_price=exit_price,
            qty=self.qty,
            pnl=net_pnl,
            close_type=close_type,
            hold_bars=self.bar_count,
        )
