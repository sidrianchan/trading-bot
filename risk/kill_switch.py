from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

# Default lives at the repo root so every entry point (systemd services,
# `main.py kill` from any CWD) agrees on the same flag file.
_DEFAULT_FLAG_FILE = Path(__file__).resolve().parent.parent / ".kill_switch"


class KillSwitch:
    """File-based emergency halt mechanism.

    Persists across process restarts. Create the flag file manually
    (touch .kill_switch at the repo root) or call trigger() to halt all
    trading immediately. Override the location with KILL_SWITCH_FILE.
    """

    def __init__(self) -> None:
        override = os.environ.get("KILL_SWITCH_FILE")
        self._flag_file = Path(override) if override else _DEFAULT_FLAG_FILE

    def is_triggered(self) -> bool:
        return self._flag_file.exists()

    def trigger(self, reason: str = "manual") -> None:
        self._flag_file.write_text(reason)
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}. All trading halted.")

    def reset(self) -> None:
        if self._flag_file.exists():
            self._flag_file.unlink()
            logger.warning("Kill switch reset. Trading may resume.")
        else:
            logger.info("Kill switch was not active.")

    def reason(self) -> str | None:
        if self._flag_file.exists():
            return self._flag_file.read_text().strip()
        return None
