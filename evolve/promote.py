"""Human decision points: approve, reject, ramp, demote.

approve_proposal() is the ONLY code path that changes live strategy behavior,
and it runs only from an explicit CLI invocation by a human.
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from evolve import lifecycle, registry
from evolve.guardrails import HARD, evolution_allowed


def approve_proposal(proposal_id: str, *, actor: str = "human", note: str = "") -> dict:
    """Promote a pending proposal: new registry version, ACTIVE pointer, capital ramp."""
    allowed, reason = evolution_allowed()
    if not allowed:
        raise RuntimeError(f"promotion refused: {reason}")

    conn = lifecycle.connect()
    try:
        p = lifecycle.get_proposal(conn, proposal_id)
        if p is None:
            raise ValueError(f"unknown proposal: {proposal_id}")
        # Validates pending_approval -> approved (raises on any other state)
        lifecycle.transition(conn, proposal_id, "approved", actor=actor, detail=note)

        strategy_id = p["strategy_id"]
        parent = registry.load_active(strategy_id)
        version = registry.next_version(strategy_id)
        record = registry.new_record(
            strategy_id, version, p["family"], p["params"],
            status="active",
            capital_fraction=HARD.new_strategy_capital_fraction,
            parent=f"{strategy_id}/v{parent['version']}" if parent else None,
            proposal_id=proposal_id,
            created_by="evolve",
            expected=p["expected"],
        )
        record["provenance"]["approved_by"] = actor
        record["provenance"]["approved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        registry.write_version(record)
        registry.set_active(strategy_id, version)
        lifecycle.transition(conn, proposal_id, "active",
                             detail=f"registry {strategy_id}/v{version} at "
                                    f"{HARD.new_strategy_capital_fraction:.0%} capital for {HARD.ramp_days}d")
        _notify_and_journal(
            f"✅ [Evolve] {proposal_id} APPROVED by {actor}: {strategy_id} -> v{version} "
            f"at {HARD.new_strategy_capital_fraction:.0%} capital (ramp {HARD.ramp_days}d). "
            f"Run `evolve ramp {strategy_id}` after the ramp period.",
            proposal_id,
        )
        return record
    finally:
        conn.close()


def reject_proposal(proposal_id: str, *, actor: str = "human", note: str = "") -> None:
    conn = lifecycle.connect()
    try:
        lifecycle.transition(conn, proposal_id, "rejected", actor=actor, detail=note)
        _notify_and_journal(f"🚫 [Evolve] {proposal_id} rejected by {actor}"
                            + (f": {note}" if note else ""), proposal_id)
    finally:
        conn.close()


def ramp_strategy(strategy_id: str, *, actor: str = "human") -> None:
    """Lift a promoted strategy to full capital after the ramp period."""
    record = registry.load_active(strategy_id)
    if record is None:
        raise ValueError(f"no active registry version for {strategy_id}")
    if record["capital_fraction"] >= 1.0:
        print(f"{strategy_id} v{record['version']} already at full capital")
        return
    registry.update_capital_fraction(strategy_id, record["version"], 1.0)
    _notify_and_journal(
        f"📈 [Evolve] {strategy_id} v{record['version']} ramped to 100% capital by {actor}",
        record["lineage"].get("proposal_id"),
    )


def demote_strategy(strategy_id: str, *, actor: str = "auto", reason: str = "") -> dict:
    """Flip ACTIVE back to the parent version. Used by the auto-demotion
    monitor and the manual `evolve demote` command."""
    record = registry.load_active(strategy_id)
    if record is None:
        raise ValueError(f"no active registry version for {strategy_id}")
    parent_ref = (record.get("lineage") or {}).get("parent")
    if not parent_ref:
        raise ValueError(f"{strategy_id} v{record['version']} has no parent to demote to")
    parent_version = int(parent_ref.rsplit("/v", 1)[1])
    registry.set_active(strategy_id, parent_version)

    proposal_id = (record.get("lineage") or {}).get("proposal_id")
    if proposal_id:
        conn = lifecycle.connect()
        try:
            p = lifecycle.get_proposal(conn, proposal_id)
            if p and p["state"] == "active":
                lifecycle.transition(conn, proposal_id, "demoted", actor=actor, detail=reason)
                lifecycle.transition(conn, proposal_id, "retired", actor=actor)
        finally:
            conn.close()

    msg = (f"⚠️ [Evolve] {strategy_id} v{record['version']} DEMOTED by {actor} "
           f"-> back to v{parent_version}" + (f" ({reason})" if reason else ""))
    _notify_and_journal(msg, proposal_id)
    logger.warning(msg)
    return registry.load_active(strategy_id) or {}


def _notify_and_journal(message: str, proposal_id: str | None) -> None:
    from monitor.notify import TelegramNotifier, journal_event

    journal_event({"event": "evolve_decision", "bot": "Evolve",
                   "proposal_id": proposal_id, "reason": message})
    TelegramNotifier().send(message.replace("[Evolve]", "<b>[Evolve]</b>"))
