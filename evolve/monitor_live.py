"""Auto-demotion monitor for LIVE strategies.

Runs inside the daily shadow-step. Compares each live strategy's drawdown
(from its state file) against its registry drawdown envelope; a promoted
version that breaches it is automatically demoted back to its parent.

Seeded v1 records (no parent) are never auto-demoted here — the existing
DrawdownMonitor and strategy circuit breakers already protect them.
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from evolve import registry
from evolve.guardrails import HARD

STATE_FILES = {
    "etf_momentum": Path("logs") / "momentum_state.json",
    "crypto_momentum": Path("logs") / "crypto_state.json",
}


def check_live_envelopes(config: dict) -> None:
    for strategy_id, state_path in STATE_FILES.items():
        try:
            _check_one(strategy_id, state_path)
        except Exception as exc:
            logger.error(f"Live envelope check failed for {strategy_id}: {exc}")


def _current_drawdown(strategy_id: str, state_path: Path) -> float | None:
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    peak = float(state.get("peak", 0.0))
    cash_value = float(state.get("cash_value") or 0.0)
    if peak <= 0:
        return None
    # State files persist peak; current scoped value is cash_value when flat.
    # When holding a position we cannot mark to market from the state file
    # alone, so fall back to the broker-free approximation: skip.
    if state.get("last_target"):
        return None
    if cash_value <= 0:
        return None
    return cash_value / peak - 1.0


def _check_one(strategy_id: str, state_path: Path) -> None:
    record = registry.load_active(strategy_id)
    if record is None:
        return
    if not (record.get("lineage") or {}).get("parent"):
        return  # seeded v1: existing risk stack owns this

    drawdown = _current_drawdown(strategy_id, state_path)
    if drawdown is None:
        return

    expected = (record.get("provenance") or {}).get("expected") or {}
    exp_dd = expected.get("max_dd")
    floors = [-HARD.demote_dd_hard]
    if isinstance(exp_dd, (int, float)) and exp_dd < 0:
        floors.append(float(exp_dd) * HARD.demote_dd_multiplier)
    envelope = max(floors)

    if drawdown <= envelope:
        from evolve.promote import demote_strategy

        logger.warning(
            f"Live envelope breach: {strategy_id} v{record['version']} "
            f"drawdown {drawdown:.1%} <= {envelope:.1%}; auto-demoting"
        )
        demote_strategy(
            strategy_id, actor="auto",
            reason=f"live drawdown {drawdown:.1%} breached envelope {envelope:.1%}",
        )
