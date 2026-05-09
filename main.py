"""Trading bot entry point.

Usage:
    python main.py status            # show current portfolio status
    python main.py kill              # trigger emergency kill switch
    python main.py reset-kill        # reset kill switch
    python main.py ta-backtest       # run the multi-TF technical-analysis backtest (Phase D)
    python main.py intraday          # live agent loop  (disabled until Phase F)
    python main.py intraday --dry-run

Legacy commands (factor strategy / VWAP fade) were removed in the TA rebuild.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"config.yaml not found at {CONFIG_PATH.resolve()}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_dir = Path(config.get("monitor", {}).get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)
    daily_dir = log_dir / "daily"
    daily_dir.mkdir(exist_ok=True)
    level = config.get("monitor", {}).get("log_level", "INFO")
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    logger.add(log_dir / "trading_bot.log", rotation="1 day", retention="30 days", level="DEBUG")
    logger.add(daily_dir / "{time:YYYY-MM-DD}.log", rotation="00:00", retention="30 days", level="INFO")


def cmd_status(config: dict) -> None:
    from execution.broker import AlpacaBroker
    from risk.kill_switch import KillSwitch

    broker = AlpacaBroker()
    ks = KillSwitch()

    value = broker.get_portfolio_value()
    cash = broker.get_cash()
    positions = broker.get_positions()

    print(f"\nPortfolio Value : ${value:>10,.2f}")
    print(f"Cash            : ${cash:>10,.2f}")
    print(f"Positions       : {len(positions)}")
    print(f"Kill Switch     : {'ACTIVE' if ks.is_triggered() else 'inactive'}")
    if not positions.empty:
        print("\nPositions:")
        print(positions.to_string())


def cmd_kill() -> None:
    from risk.kill_switch import KillSwitch
    reason = input("Reason for kill switch (Enter to confirm): ").strip() or "manual trigger"
    KillSwitch().trigger(reason)


def cmd_reset_kill() -> None:
    from risk.kill_switch import KillSwitch
    KillSwitch().reset()


def cmd_intraday(config: dict, dry_run: bool = False) -> None:
    logger.error(
        "Live intraday loop is disabled until Phase F. "
        "Run `python main.py ta-backtest` first; the gate must pass before paper trading resumes."
    )
    sys.exit(2)


def cmd_ta_backtest(config: dict) -> None:
    try:
        from backtester.ta_engine import run_ta_backtest  # type: ignore
    except ImportError:
        logger.error("TA backtester not implemented yet — arrives in Phase D.")
        sys.exit(2)
    run_ta_backtest(config)


def cli() -> None:
    config = load_config()
    setup_logging(config)

    dry_run = "--dry-run" in sys.argv

    commands = {
        "status":       lambda: cmd_status(config),
        "kill":         cmd_kill,
        "reset-kill":   cmd_reset_kill,
        "intraday":     lambda: cmd_intraday(config, dry_run=dry_run),
        "ta-backtest":  lambda: cmd_ta_backtest(config),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(0)

    commands[sys.argv[1]]()


if __name__ == "__main__":
    cli()
