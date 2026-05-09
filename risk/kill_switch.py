from __future__ import annotations

from pathlib import Path

from loguru import logger

_FLAG_FILE = Path(".kill_switch")


class KillSwitch:
    """File-based emergency halt mechanism.

    Persists across process restarts. Create the flag file manually
    (touch .kill_switch) or call trigger() to halt all trading immediately.
    """

    def is_triggered(self) -> bool:
        return _FLAG_FILE.exists()

    def trigger(self, reason: str = "manual") -> None:
        _FLAG_FILE.write_text(reason)
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}. All trading halted.")

    def reset(self) -> None:
        if _FLAG_FILE.exists():
            _FLAG_FILE.unlink()
            logger.warning("Kill switch reset. Trading may resume.")
        else:
            logger.info("Kill switch was not active.")

    def reason(self) -> str | None:
        if _FLAG_FILE.exists():
            return _FLAG_FILE.read_text().strip()
        return None
