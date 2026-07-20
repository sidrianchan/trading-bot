from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from loguru import logger

from execution.broker import AlpacaBroker
from execution.orders import Side
from monitor.notify import TelegramNotifier
from risk.kill_switch import KillSwitch
from signals.crypto_momentum import (
    CryptoMomentumConfig,
    CryptoMomentumState,
    alpaca_to_yahoo_symbol,
    compute_crypto_signal,
    normalize_crypto_symbol,
    state_from_dict,
    state_to_dict,
)


def _guarded(fn):
    """Wrap a scheduled job so an exception can't escape schedule.run_pending()
    and kill the 24/7 loop; a failure logs and skips this cycle instead."""
    def wrapper():
        try:
            fn()
        except Exception as exc:
            logger.error(f"Scheduled {fn.__name__} failed (will retry next cycle): {exc}", exc_info=True)
    return wrapper


class CryptoPaperLoop:
    """24/7 paper loop for the BTC/ETH momentum strategy."""

    def __init__(self, app_config: dict, broker: AlpacaBroker | None = None):
        self.app_config = app_config
        self.crypto_config = self._build_config(app_config.get("crypto", {}))
        self.broker = broker
        self.state_path = Path("logs") / "crypto_state.json"
        self.daily_log_dir = Path("logs") / "crypto_daily"
        self.state_path.parent.mkdir(exist_ok=True)
        self.daily_log_dir.mkdir(parents=True, exist_ok=True)
        self.et = ZoneInfo("America/New_York")
        self.order_cfg = app_config.get("crypto", {}).get("order", {})
        self.cancel_after_minutes = int(self.order_cfg.get("cancel_after_minutes", 5))
        self.mid_tolerance = float(self.order_cfg.get("mid_price_tolerance", 0.001))
        self.max_order_attempts = int(self.order_cfg.get("max_attempts", 3))
        self._stable_tradable: bool | None = None
        self.notify = TelegramNotifier()

    @staticmethod
    def _yaml_config(config: dict, capital: float) -> CryptoMomentumConfig:
        abs_cfg = config.get("absolute_momentum", {})
        rel_cfg = config.get("relative_momentum", {})
        cb_cfg = config.get("circuit_breaker", {})
        return CryptoMomentumConfig(
            capital=capital,
            universe=tuple(config.get("universe", ["BTC/USD", "ETH/USD"])),
            stable=str(config.get("stable", "USDC/USD")),
            abs_lookback=int(abs_cfg.get("lookback_days", 84)),
            abs_skip=int(abs_cfg.get("skip_days", 14)),
            rel_lookback=int(rel_cfg.get("lookback_days", 7)),
            rel_skip=int(rel_cfg.get("skip_days", 14)),
            cb_threshold=float(cb_cfg.get("max_drawdown_from_peak", -0.40)),
        )

    @staticmethod
    def _build_config(config: dict) -> CryptoMomentumConfig:
        from evolve import registry

        # The registry is the source of truth for which strategy version runs
        # live. Capital is a runtime concern (config.yaml x the promotion ramp),
        # not an evolvable parameter, so it is passed as an override rather than
        # stored in the registry record. resolve_config falls back to the
        # config.yaml values on any registry failure.
        capital = float(config.get("capital", 30_000.0)) * registry.capital_fraction(
            "crypto_momentum"
        )
        return registry.resolve_config(
            "crypto_momentum",
            CryptoPaperLoop._yaml_config(config, capital),
            overrides={"capital": capital},
        )

    def load_state(self) -> CryptoMomentumState:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return state_from_dict(json.load(f))
        return CryptoMomentumState(peak=self.crypto_config.capital, cash_value=self.crypto_config.capital)

    def save_state(self, state: CryptoMomentumState) -> None:
        with open(self.state_path, "w") as f:
            json.dump(state_to_dict(state), f, indent=2)

    def fetch_history(self) -> pd.DataFrame:
        max_history = max(
            self.crypto_config.abs_lookback + self.crypto_config.abs_skip,
            self.crypto_config.rel_lookback + self.crypto_config.rel_skip,
        )
        end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=max(220, max_history + 30))
        tickers = [alpaca_to_yahoo_symbol(symbol) for symbol in self.crypto_config.universe]
        raw = yf.download(tickers, start=start.date(), end=end.date(), auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            raise RuntimeError("No BTC/ETH yfinance data returned")
        prices = raw["Close"].rename(columns={"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"})
        return prices.dropna(how="all").ffill().dropna()

    def scoped_value(self, state: CryptoMomentumState) -> float:
        if self.broker is None:
            return state.cash_value or self.crypto_config.capital
        positions = self.broker.get_positions_for_symbols(list(self.crypto_config.universe) + [self.crypto_config.stable])
        position_value = 0.0 if positions.empty else float(positions["market_value"].sum())
        if position_value > 0:
            stable_norm = normalize_crypto_symbol(self.crypto_config.stable)
            if all(idx == stable_norm for idx in positions.index):
                # Holding stable only — state.cash_value would double-count the position
                return position_value
            return position_value + max(0.0, state.cash_value)
        return state.cash_value or self.crypto_config.capital

    def dry_run(self) -> None:
        state = CryptoMomentumState(peak=self.crypto_config.capital, cash_value=self.crypto_config.capital)
        prices = self.fetch_history()
        signal, _ = compute_crypto_signal(prices, state, self.crypto_config.capital, self.crypto_config)
        self.print_signal(signal, dry=True)

    def rebalance_once(self, dry: bool = False) -> None:
        if KillSwitch().is_triggered():
            logger.critical("Kill switch is ACTIVE. Skipping crypto rebalance.")
            return
        if dry:
            self.dry_run()
            return
        self._ensure_broker()
        state = self.load_state()
        prices = self.fetch_history()
        self._seed_btc_benchmark(state, prices)
        current_value = self.scoped_value(state)
        signal, new_state = compute_crypto_signal(prices, state, current_value, self.crypto_config)
        self.print_signal(signal, dry=False)
        prev_executed = state.last_executed_target
        executed = self._execute_target(signal.target, current_value, new_state)
        if executed:
            new_state.last_executed_target = signal.target
            if signal.target != prev_executed:
                self.notify.rebalance(
                    "Crypto", signal.target or "USDC", signal.regime, current_value, signal.decision_reason
                )
        else:
            logger.error(
                f"Crypto rebalance execution failed; position unchanged "
                f"(target={signal.target or 'USDC'}, holding={prev_executed or 'USDC'})"
            )
        new_state.last_eval_date = datetime.now(tz=self.et).date().isoformat()
        new_state.cash_value = self.scoped_value(new_state) if signal.target is None else 0.0
        self.save_state(new_state)

    def circuit_check(self) -> None:
        if KillSwitch().is_triggered():
            logger.critical("Kill switch is ACTIVE. Skipping crypto circuit check.")
            return
        self._ensure_broker()
        state = self.load_state()
        current_value = self.scoped_value(state)
        if state.peak <= 0:
            state.peak = self.crypto_config.capital
        state.peak = max(state.peak, current_value)
        drawdown = current_value / state.peak - 1.0 if state.peak > 0 else 0.0
        risk_positions = self.broker.get_positions_for_symbols(self.crypto_config.universe)
        if not risk_positions.empty and drawdown <= self.crypto_config.cb_threshold:
            logger.warning(
                f"Crypto circuit breaker triggered: {drawdown:.1%} <= "
                f"{self.crypto_config.cb_threshold:.0%}"
            )
            self._sell_positions(risk_positions)
            state.cash_value = current_value
            state.last_target = None
            state.last_executed_target = None
            self.notify.circuit_breaker("Crypto", drawdown, current_value)
        self.save_state(state)

    def write_daily_summary(self) -> None:
        state = self.load_state()
        prices = self.fetch_history()
        self._seed_btc_benchmark(state, prices)
        current_value = self.scoped_value(state)
        signal, state = compute_crypto_signal(prices, state, current_value, self.crypto_config)
        state.last_eval_date = datetime.now(tz=self.et).date().isoformat()
        self.save_state(state)

        latest_btc = float(prices["BTC/USD"].iloc[-1])
        btc_benchmark = None
        if state.start_btc_price and state.start_btc_price > 0:
            btc_benchmark = self.crypto_config.capital * latest_btc / float(state.start_btc_price)
        benchmark_line = "BTC benchmark: n/a"
        if btc_benchmark is not None:
            benchmark_line = f"BTC benchmark: ${btc_benchmark:,.2f}; P&L vs BTC: ${current_value - btc_benchmark:,.2f}"

        holding = signal.target or "USDC"
        lines = [
            f"Crypto daily summary {datetime.now(tz=self.et).date().isoformat()}",
            f"Portfolio value: ${current_value:,.2f} vs ${self.crypto_config.capital:,.2f} starting capital",
            f"Current holding: {holding}",
            f"BTC {self.crypto_config.abs_lookback}d momentum: {signal.btc_abs_return:+.2%}",
            "Relative momentum: "
            f"BTC {signal.relative_scores.get('BTC/USD', float('nan')):+.2%}; "
            f"ETH {signal.relative_scores.get('ETH/USD', float('nan')):+.2%}",
            benchmark_line,
            f"Decision: {signal.decision_reason}",
        ]
        log_path = self.daily_log_dir / f"{datetime.now(tz=self.et).date().isoformat()}.log"
        log_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Crypto daily summary written to {log_path}")
        self.notify.health("Crypto", current_value, holding, signal.decision_reason)

    def run(self) -> None:
        import schedule

        self._ensure_broker()
        if not self.broker.is_paper:
            raise RuntimeError("BROKER IS NOT IN PAPER MODE. ABORTING.")
        logger.info("Starting crypto paper loop. Schedule: Monday 09:00 ET rebalance, hourly circuit check, daily 08:00 ET summary")
        self.notify.startup("Crypto")
        _guarded(self.circuit_check)()
        schedule.every().monday.at("09:00", "America/New_York").do(_guarded(self.rebalance_once))
        schedule.every().hour.do(_guarded(self.circuit_check))
        schedule.every().day.at("08:00", "America/New_York").do(_guarded(self.write_daily_summary))
        while True:
            schedule.run_pending()
            time.sleep(30)

    def print_signal(self, signal, dry: bool) -> None:
        print(f"\n{'=' * 72}")
        print(f"  BTC/ETH CRYPTO MOMENTUM - {'DRY RUN' if dry else 'PAPER'}")
        print(f"{'=' * 72}")
        print(f"  Capital base        : ${self.crypto_config.capital:>12,.2f}")
        print(f"  BTC 84d momentum    : {signal.btc_abs_return:>12.2%}")
        print(f"  Regime              : {signal.regime:>12}")
        print(f"  Drawdown from peak  : {signal.drawdown:>12.2%}")
        print(f"  CB threshold        : {self.crypto_config.cb_threshold:>12.0%}")
        print("  Relative momentum:")
        for symbol, value in sorted(signal.relative_scores.items(), key=lambda item: item[0]):
            marker = " <- TARGET" if symbol == signal.target else ""
            print(f"     {symbol:<8} {value:>+8.2%}{marker}")
        print(f"  Target              : {signal.target or 'USDC':>12}")
        print(f"  Reason              : {signal.decision_reason}")
        if dry:
            print("\n  [dry-run] No orders submitted, no state file modified.")
        print(f"{'=' * 72}\n")

    def _execute_target(self, target: str | None, current_value: float, state: CryptoMomentumState) -> bool:
        self._ensure_broker()
        positions = self.broker.get_positions_for_symbols(list(self.crypto_config.universe) + [self.crypto_config.stable])
        stable_target = self.crypto_config.stable if target is None and self._is_stable_tradable() else None
        desired = target or stable_target

        if not positions.empty:
            to_sell = positions[positions["normalized_symbol"] != desired] if desired else positions
            self._sell_positions(to_sell)

        if desired is None:
            state.cash_value = current_value
            return True

        refreshed = self.broker.get_positions_for_symbols([desired])
        position_mv = float(refreshed["market_value"].sum()) if not refreshed.empty else 0.0
        if position_mv >= current_value * 0.98:
            logger.info(f"Crypto already positioned in {desired}; no buy needed")
            state.cash_value = 0.0 if desired in self.crypto_config.universe else current_value
            return True
        if desired == self.crypto_config.stable:
            # Parking: USD cash is economically equivalent to the stable, so skip
            # the buy when stable position + free cash already covers scope.
            # Covers Alpaca paper reporting USDC as cash, and residual dust that
            # leaves the position just under the 98% line — a top-off limit order
            # at 1.00 won't fill and just burns the retry budget.
            free_cash = min(self.broker.get_cash(), self.crypto_config.capital)
            if position_mv + free_cash >= current_value * 0.98:
                logger.info(f"Crypto parked: {desired} position + cash covers scope; no top-off buy")
                state.cash_value = current_value
                return True

        # Cap to this bot's configured scope so it never spends the ETF bot's cash.
        available_cash = min(self.broker.get_cash(), self.crypto_config.capital)
        notional = min(current_value, available_cash)
        if notional < 1.0:
            logger.warning(f"Insufficient cash for crypto buy: ${available_cash:,.2f}")
            return False
        if self._submit_limit_until_filled(desired, Side.BUY, notional=notional):
            state.cash_value = 0.0 if desired in self.crypto_config.universe else current_value
            return True
        return False

    def _sell_positions(self, positions: pd.DataFrame) -> None:
        if positions.empty:
            return
        for _, pos in positions.iterrows():
            symbol = str(pos["normalized_symbol"])
            qty = abs(float(pos["qty"]))
            if qty <= 0:
                continue
            self._submit_limit_until_filled(symbol, Side.SELL, qty=qty)

    def _submit_limit_until_filled(
        self,
        symbol: str,
        side: Side,
        notional: float | None = None,
        qty: float | None = None,
    ) -> bool:
        self._ensure_broker()
        for attempt in range(1, self.max_order_attempts + 1):
            mid = self.broker.get_crypto_mid_price(symbol)
            order_id = self.broker.submit_crypto_limit_order(symbol, side, mid, notional=notional, qty=qty)
            if not order_id:
                return False
            order = self.broker.wait_for_order_fill(order_id, timeout_seconds=self.cancel_after_minutes * 60)
            status = str(getattr(order, "status", "")).lower() if order is not None else "unknown"
            if status == "filled":
                fill_price = float(getattr(order, "filled_avg_price", mid) or mid)
                filled_qty = float(getattr(order, "filled_qty", 0) or 0)
                value = fill_price * filled_qty if filled_qty else (notional or 0.0)
                deviation = abs(fill_price - mid) / mid
                if deviation > self.mid_tolerance:
                    logger.error(
                        f"Crypto fill slippage exceeded tolerance for {symbol}: "
                        f"fill={fill_price:.2f} mid={mid:.2f} deviation={deviation:.3%}"
                    )
                self.notify.trade("Crypto", side.value.upper(), symbol, value,
                                  f"fill@{fill_price:.2f}")
                return True
            cancelled = self.broker.cancel_order(order_id)
            if not cancelled:
                # Order filled between the timeout check and the cancel call.
                order = self.broker.get_order(order_id)
                fill_price = float(getattr(order, "filled_avg_price", mid) or mid)
                filled_qty = float(getattr(order, "filled_qty", 0) or 0)
                value = fill_price * filled_qty if filled_qty else (notional or 0.0)
                self.notify.trade("Crypto", side.value.upper(), symbol, value, f"fill@{fill_price:.2f}")
                return True
            # The order may have partially filled before we cancelled the remainder.
            # If the resulting position already covers our scope, treat it as success
            # so we set last_executed_target instead of re-entering the full buy path.
            if side == Side.BUY:
                filled = self.broker.get_positions_for_symbols([symbol])
                position_value = 0.0 if filled.empty else float(filled["market_value"].sum())
                if position_value >= self.crypto_config.capital * 0.95:
                    logger.info(
                        f"Crypto {symbol} position ${position_value:,.2f} already >= 95% of "
                        f"capital (${self.crypto_config.capital:,.2f}) after partial fill; "
                        f"treating buy as complete"
                    )
                    self.notify.trade("Crypto", side.value.upper(), symbol, position_value,
                                      "partial fill settled to target")
                    return True
            logger.warning(f"Crypto order unfilled after {self.cancel_after_minutes}m; resubmitting attempt {attempt + 1}")
        logger.error(f"Crypto order failed after {self.max_order_attempts} attempts: {side.value} {symbol}")
        return False

    def _is_stable_tradable(self) -> bool:
        if self._stable_tradable is not None:
            return self._stable_tradable
        self._ensure_broker()
        try:
            asset = self.broker.get_crypto_asset_metadata(self.crypto_config.stable)
            self._stable_tradable = bool(getattr(asset, "tradable", False))
        except Exception as exc:
            logger.warning(f"Stable pair {self.crypto_config.stable} unavailable; using USD cash: {exc}")
            self._stable_tradable = False
        return self._stable_tradable

    def _seed_btc_benchmark(self, state: CryptoMomentumState, prices: pd.DataFrame) -> None:
        if state.start_btc_price is None and not prices.empty:
            state.start_btc_price = float(prices["BTC/USD"].iloc[-1])

    def _ensure_broker(self) -> None:
        if self.broker is None:
            self.broker = AlpacaBroker()
