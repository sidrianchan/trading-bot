"""Live intraday agent loop.

Phase A status: stubbed. The kill-switch and drawdown wiring is preserved
intact so safety controls work the moment the loop is reactivated. The
TA-driven scan / trigger / execution pipeline is rebuilt in Phase F after the
backtest gate passes.

Until Phase F lands, calling any of the scheduled methods raises NotImplementedError
so a misconfigured cron cannot accidentally run a half-built loop in paper mode.
"""
from __future__ import annotations

from datetime import date

from loguru import logger

from execution import AlpacaBroker
from risk import KillSwitch, DrawdownMonitor


_PHASE_F_PENDING = (
    "Live agent loop is disabled until Phase F (post-backtest-gate). "
    "Run `python main.py ta-backtest` to validate the TA stack first."
)


class IntradayAgentLoop:
    """Stubbed orchestrator — wraps the kill switch and broker only.

    Real scheduling (`run_morning_scan`, `start_streaming`, `classify_day`,
    `hard_close`, `run_report`) is intentionally inert during the rebuild.
    """

    def __init__(self, config: dict):
        self.config = config
        risk_cfg = config.get("risk", {})
        intra_risk_cfg = config.get("ta", {}).get("risk", {})

        self.kill_switch = KillSwitch()
        self.drawdown = DrawdownMonitor(
            limit=risk_cfg.get("portfolio_drawdown_limit", 0.15),
            reset_threshold=risk_cfg.get("drawdown_reset_threshold", 0.07),
            daily_pnl_halt_pct=intra_risk_cfg.get("daily_pnl_halt_pct", -0.02),
        )
        self.broker = AlpacaBroker()
        logger.info("IntradayAgentLoop initialized (Phase A stub — live methods disabled)")

    # ------------------------------------------------------------------
    # Kill-switch / safety surface (preserved across the rebuild)
    # ------------------------------------------------------------------

    def is_safe_to_trade(self) -> bool:
        if self.kill_switch.is_triggered():
            logger.critical(f"Kill switch active: {self.kill_switch.reason()}")
            return False
        if self.drawdown.is_halted:
            logger.critical("Portfolio drawdown halt active")
            return False
        return True

    # ------------------------------------------------------------------
    # Scheduled hooks — disabled until Phase F
    # ------------------------------------------------------------------

    def run_morning_scan(self) -> None:
        raise NotImplementedError(_PHASE_F_PENDING)

    def start_streaming(self) -> None:
        raise NotImplementedError(_PHASE_F_PENDING)

    def classify_day(self) -> None:
        raise NotImplementedError(_PHASE_F_PENDING)

    def hard_close(self) -> None:
        # Hard close is genuinely safety-critical, so route it through the
        # broker even when the rest of the loop is stubbed.
        logger.warning("Stub hard_close — liquidating any open intraday positions")
        try:
            self.broker.market_sell_all_intraday()
        except Exception as exc:
            logger.error(f"Hard-close fallback failed: {exc}")

    def run_report(self) -> None:
        logger.info(f"{date.today()}: report stub (Phase F pending)")
