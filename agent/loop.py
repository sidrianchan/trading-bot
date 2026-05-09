from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

from data.universe import get_intraday_universe, apply_liquidity_filter
from data.streaming import BarStreamer, build_snapshots_from_alpaca
from signals.intraday_composite import IntradayComposite
from signals.gap import StockSnapshot
from risk import KillSwitch, DrawdownMonitor, RiskLimits
from execution import AlpacaBroker, generate_rebalance_orders
from execution.orders import BracketOrder, Side
from portfolio.position_manager import PositionManager
from monitor import DailyReporter

_PAPER_STATE_FILE = Path("logs/paper_state.json")

# Hard-coded — not configurable. No overnight positions, ever.
_NO_OVERNIGHT = True


class IntradayAgentLoop:
    """Intraday trading orchestrator.

    Daily schedule:
      09:25 ET — universe scan, gap screen, candidate selection
      09:30 ET — start streaming 1-min bars for top candidates
      09:45 ET — classify day type (trending / range-bound)
      09:45–15:45 — active trading: ORB + VWAP signal evaluation
      15:45 ET — hard close all positions
      16:00 ET — daily summary to log

    No positions are carried overnight under any circumstances.
    """

    def __init__(self, config: dict):
        self.config = config
        intra = config.get("intraday", {})
        risk_cfg   = intra.get("risk", {})
        sig_cfg    = intra.get("signals", {})
        cand_cfg   = intra.get("candidates", {})
        uni_cfg    = intra.get("universe", {})
        monitor_cfg = config.get("monitor", {})
        bt_cfg     = config.get("backtest", {})

        initial_capital = bt_cfg.get("initial_capital", 100_000.0)

        self.kill_switch = KillSwitch()
        self.drawdown = DrawdownMonitor(
            limit=config.get("risk", {}).get("portfolio_drawdown_limit", 0.15),
            reset_threshold=config.get("risk", {}).get("drawdown_reset_threshold", 0.07),
            daily_pnl_halt_pct=risk_cfg.get("daily_pnl_halt_pct", -0.02),
        )
        self.risk = RiskLimits(
            max_position_size=config.get("portfolio", {}).get("max_position_size", 0.10),
            max_single_trade_risk=risk_cfg.get("risk_per_trade_pct", 0.005),
        )
        self.broker = AlpacaBroker()
        self.streamer = BarStreamer()
        self.reporter = DailyReporter(
            initial_capital=initial_capital,
            log_dir=monitor_cfg.get("log_dir", "logs"),
        )
        self.composite = IntradayComposite(
            top_n=cand_cfg.get("top_n", 15),
            gap_min_pct=sig_cfg.get("gap_min_pct", 1.5),
            volume_ratio_min=sig_cfg.get("premarket_volume_ratio", 2.0),
            atr_min_dollars=sig_cfg.get("atr_min_dollars", 0.50),
            orb_minutes=sig_cfg.get("orb_minutes", 15),
            vwap_std_threshold=sig_cfg.get("vwap_std_threshold", 1.5),
            vwap_min_dollar_deviation_pct=sig_cfg.get("vwap_min_dollar_deviation_pct", 0.015),
            vwap_confirm_reversal=sig_cfg.get("vwap_confirm_reversal", True),
        )
        self.position_manager = PositionManager(
            broker=self.broker,
            max_concurrent=risk_cfg.get("max_concurrent_positions", 8),
        )

        self._uni_sources = uni_cfg.get("sources", ["sp500", "russell1000"])
        self._min_adv_usd = uni_cfg.get("min_adv_usd", 10_000_000)
        self._min_price   = uni_cfg.get("min_price", 5.0)
        self._risk_pct    = risk_cfg.get("risk_per_trade_pct", 0.005)
        self._vwap_stop_std = sig_cfg.get("vwap_stop_std", 2.5)
        self._trend_size_mult = risk_cfg.get("trend_day_size_mult", 0.5)
        self._min_rr_ratio = risk_cfg.get("min_rr_ratio", 1.5)
        self._day_trend_threshold = sig_cfg.get("day_trend_threshold", 0.01)
        cooldown_bars = risk_cfg.get("cooldown_bars_after_stop", 15)
        self._cooldown = timedelta(minutes=cooldown_bars)   # 1-min bars
        self._max_trades_per_ticker = risk_cfg.get("max_trades_per_ticker_per_day", 2)
        self._daily_start_value: float = 0.0
        self._spy_prev_close: float | None = None

    # ------------------------------------------------------------------
    # 09:25 ET — Morning scan
    # ------------------------------------------------------------------

    def run_morning_scan(self) -> None:
        today = date.today()
        logger.info(f"Morning scan starting — {today}")

        if self.kill_switch.is_triggered():
            logger.critical(f"Kill switch active ({self.kill_switch.reason()}). Aborting.")
            return

        # Liquidate any stale positions from a previous crash/restart
        if _NO_OVERNIGHT:
            n = self.broker.market_sell_all_intraday()
            if n:
                logger.warning(f"Startup liquidation: closed {n} stale overnight position(s)")

        portfolio_value = self.broker.get_portfolio_value()
        self._daily_start_value = portfolio_value
        self.drawdown.update(portfolio_value)
        self.drawdown.reset_daily(portfolio_value)

        self.composite.reset()
        self.position_manager.reset_day()

        # Build universe and fetch snapshots
        tickers = get_intraday_universe(self._uni_sources)
        logger.info(f"Universe: {len(tickers)} tickers")

        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        raw_snaps  = build_snapshots_from_alpaca(tickers, api_key, secret_key)
        snaps      = apply_liquidity_filter(raw_snaps, self._min_adv_usd, self._min_price)

        # Store SPY's pre-market price for day-type classification at 09:45
        spy_snap = next((s for s in raw_snaps if s.ticker == "SPY"), None)
        if spy_snap:
            self._spy_prev_close = spy_snap.prev_close

        # Gap screen + candidate selection
        candidates = self.composite.rank_candidates(snaps)
        if not candidates:
            logger.warning("No gap candidates today — will observe only")
            return

        # Subscribe to bars for all candidates + SPY
        watch_list = [c.ticker for c in candidates] + ["SPY"]
        self.streamer.subscribe(watch_list, callback=self._on_bar)
        logger.info(f"Watchlist: {[c.ticker for c in candidates]}")

    # ------------------------------------------------------------------
    # 09:30 ET — Start streaming
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        self.streamer.start()
        logger.info(f"Streaming started ({self.streamer.mode} mode)")

    # ------------------------------------------------------------------
    # 09:45 ET — Classify day type
    # ------------------------------------------------------------------

    def classify_day(self) -> None:
        if self._spy_prev_close is None:
            self.composite.set_day_type("trending")
            return

        try:
            positions = self.broker.get_positions()
            # Approximate SPY current price from broker snapshot
            snaps = self.broker.get_snapshots(["SPY"])
            spy_now = snaps[0].latest_price if snaps else self._spy_prev_close
        except Exception:
            spy_now = self._spy_prev_close

        move = abs(spy_now / self._spy_prev_close - 1) if self._spy_prev_close else 0
        day_type = "trending" if move >= self._day_trend_threshold else "range"
        self.composite.set_day_type(day_type)
        logger.info(f"SPY move: {move:.2%} → day type: {day_type}")

    # ------------------------------------------------------------------
    # 09:45–15:45 — Per-bar callback (from streaming thread)
    # ------------------------------------------------------------------

    def _on_bar(self, ticker: str, bar: dict) -> None:
        if self.kill_switch.is_triggered():
            return

        # Delegate bar to position manager (partial exit tracking)
        self.position_manager.on_bar(ticker, bar)

        # Check intraday daily P&L halt
        portfolio_value = self._estimate_portfolio_value(bar["close"])
        self.drawdown.update_intraday(portfolio_value)
        if self.drawdown.daily_pnl_halted or self.drawdown.is_halted:
            return

        # Route bar through composite signal
        signal = self.composite.on_bar(ticker, bar)
        if signal is None:
            return

        self._try_enter(ticker, signal, bar)

    def _try_enter(self, ticker: str, signal: str, bar: dict) -> None:
        if self.position_manager.has_position(ticker):
            return
        if signal not in ("fade_long", "fade_short"):
            return

        # Per-ticker daily cap and post-stop cooldown — derived from
        # PositionManager.closed_today (single source of truth, no duplicate state)
        closed = self.position_manager.closed_today
        opened_today = sum(1 for c in closed if c["ticker"] == ticker) + (
            1 if self.position_manager.has_position(ticker) else 0
        )
        if opened_today >= self._max_trades_per_ticker:
            return
        last_stop_at = max(
            (c.get("closed_at") for c in closed
             if c["ticker"] == ticker and c.get("close_type") == "stop"
             and c.get("closed_at") is not None),
            default=None,
        )
        if last_stop_at is not None and datetime.now() - last_stop_at < self._cooldown:
            return

        price = bar["close"]
        portfolio_value = self.broker.get_portfolio_value()

        candidate = next(
            (c for c in self.composite.get_candidates() if c.ticker == ticker), None
        )
        if candidate is None:
            return

        vwap = self.composite.get_vwap(ticker)
        std  = self.composite.get_vwap_std(ticker)
        if vwap is None or std is None or std <= 0:
            return

        stop_band = self._vwap_stop_std * std
        if signal == "fade_long":
            if price >= vwap:
                return
            side = Side.BUY
            entry  = round(price, 2)
            stop   = round(vwap - stop_band, 2)
            target = round(vwap, 2)
        else:  # fade_short
            if price <= vwap:
                return
            side = Side.SELL
            entry  = round(price, 2)
            stop   = round(vwap + stop_band, 2)
            target = round(vwap, 2)

        risk_per_share   = abs(entry - stop)
        reward_per_share = abs(target - entry)
        if risk_per_share < 0.01 or reward_per_share <= 0:
            return
        if reward_per_share / risk_per_share < self._min_rr_ratio:
            return

        size_mult = self._trend_size_mult if self.composite.get_day_type() == "trend" else 1.0
        if size_mult <= 0:
            return

        base_qty = self.risk.size_from_risk(portfolio_value, entry, stop, self._risk_pct)
        qty = max(1, int(base_qty * size_mult)) if base_qty > 0 else 0
        if qty <= 0:
            return

        order = BracketOrder(
            ticker=ticker,
            side=side,
            qty=qty,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason=f"VWAP {signal} (day={self.composite.get_day_type()})",
        )
        self.position_manager.try_enter(order)

    # ------------------------------------------------------------------
    # 15:45 ET — Hard close
    # ------------------------------------------------------------------

    def hard_close(self) -> None:
        logger.warning("15:45 ET — initiating hard close of all intraday positions")
        self.streamer.stop()
        self.position_manager.hard_close_all()
        logger.info("Hard close complete. No overnight positions.")

    # ------------------------------------------------------------------
    # 16:00 ET — Daily report
    # ------------------------------------------------------------------

    def run_report(self) -> None:
        portfolio_value = self.broker.get_portfolio_value()
        benchmark_value = self._benchmark_value(date.today())

        self.reporter.intraday_summary(
            trades=self.position_manager.closed_today,
            portfolio_value=portfolio_value,
            benchmark_value=benchmark_value,
            daily_start_value=self._daily_start_value,
            candidates=self.composite.get_candidates(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_portfolio_value(self, latest_price: float) -> float:
        try:
            return self.broker.get_portfolio_value()
        except Exception:
            return self._daily_start_value

    def _benchmark_value(self, today: date) -> float | None:
        state = self._load_paper_state(today)
        if state is None:
            return None
        spy_start = state["spy_start_price"]
        snaps = self.broker.get_snapshots(["SPY"])
        if not snaps:
            return None
        spy_now = snaps[0].latest_price
        return self.reporter.initial_capital * (spy_now / spy_start) if spy_start > 0 else None

    def _load_paper_state(self, today: date) -> dict | None:
        if _PAPER_STATE_FILE.exists():
            return json.loads(_PAPER_STATE_FILE.read_text())
        try:
            snaps = self.broker.get_snapshots(["SPY"])
            spy_start = snaps[0].latest_price if snaps else 0
        except Exception:
            return None
        state = {"start_date": str(today), "spy_start_price": spy_start}
        _PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PAPER_STATE_FILE.write_text(json.dumps(state, indent=2))
        return state
