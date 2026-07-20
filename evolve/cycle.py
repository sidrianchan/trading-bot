"""Weekly evolution cycle orchestration.

digest -> LLM proposals -> guardrails -> backtest gate + holdout -> shadow.
Promotion beyond shadow requires the daily shadow job (evolve/shadow.py) and
an explicit human approval — never this module.
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from evolve import lifecycle, llm, registry
from evolve.evidence import build_digest
from evolve.families import get_family
from evolve.guardrails import HARD, evolution_allowed


def run_cycle(config: dict, dry_run: bool = False) -> None:
    allowed, reason = evolution_allowed()
    if not allowed:
        print(f"Evolution cycle refused: {reason}")
        return

    digest = build_digest(config)
    if dry_run:
        print("=" * 72)
        print("EVOLVE DRY RUN — no LLM call, no state changes")
        print("=" * 72)
        print("\n----- SYSTEM PROMPT -----\n")
        print(llm.system_prompt())
        print("\n----- DIGEST (user turn) -----\n")
        print(digest)
        return

    conn = lifecycle.connect()
    try:
        _run_cycle_inner(config, conn, digest)
    finally:
        conn.close()


def _run_cycle_inner(config: dict, conn, digest: str) -> None:
    from monitor.notify import TelegramNotifier

    notify = TelegramNotifier()
    cycle_id = f"c-{datetime.now(timezone.utc).date().isoformat()}"

    try:
        batch, call_id = llm.propose(digest, config, conn)
    except llm.ProposalParseError as exc:
        logger.error(f"Evolve cycle {cycle_id}: unparseable LLM reply: {exc}")
        notify.send(f"⚠️ <b>[Evolve]</b> Cycle {cycle_id} aborted: unparseable LLM reply")
        return
    except Exception as exc:
        logger.error(f"Evolve cycle {cycle_id}: LLM call failed: {exc}")
        notify.send(f"⚠️ <b>[Evolve]</b> Cycle {cycle_id} aborted: LLM call failed ({exc})")
        return

    proposals = batch["proposals"][: HARD.max_candidates_per_cycle]
    summary = [f"🧬 <b>[Evolve]</b> Cycle {cycle_id}",
               f"Analysis: {batch.get('analysis_summary', '')[:400]}"]
    if not proposals:
        summary.append("No candidates proposed this cycle.")
        logger.info(f"Evolve cycle {cycle_id}: no proposals")
        notify.send("\n".join(summary))
        return

    for p in proposals:
        outcome = _process_proposal(conn, cycle_id, call_id, p)
        summary.append(f"• {outcome}")

    notify.send("\n".join(summary))


def _process_proposal(conn, cycle_id: str, call_id: int, p: dict) -> str:
    """Take one proposal through guardrails and validation. Returns a summary line."""
    pid = lifecycle.new_proposal_id()
    family = get_family(p["family"])
    lifecycle.insert_proposal(
        conn, proposal_id=pid, cycle_id=cycle_id, strategy_id=p["strategy_id"],
        family=p["family"], params=p["params"], hypothesis=p["hypothesis"],
        expected=p["expected"], llm_call_id=call_id,
    )

    violations = family.validate_params(p["params"])
    if violations:
        lifecycle.transition(conn, pid, "rejected_guardrail", detail="; ".join(violations))
        return f"{pid} {p['strategy_id']}: rejected by guardrails ({violations[0]})"

    n_shadows = lifecycle.count_in_states(conn, {"shadow"})
    if n_shadows >= HARD.max_concurrent_shadows:
        lifecycle.transition(
            conn, pid, "rejected_guardrail",
            detail=f"max concurrent shadows ({HARD.max_concurrent_shadows}) reached",
        )
        return f"{pid} {p['strategy_id']}: rejected (shadow slots full)"

    from evolve.validator import validate_candidate

    active = registry.load_active(p["strategy_id"])
    incumbent_params = active["params"] if active else family.default_params()
    n_trials = lifecycle.trial_count(conn, p["family"])
    lifecycle.bump_trials(conn, p["family"])

    try:
        result = validate_candidate(p["family"], p["params"], incumbent_params, n_trials)
    except Exception as exc:
        logger.error(f"Evolve validation failed for {pid}: {exc}")
        lifecycle.transition(conn, pid, "backtest_fail", detail=f"validation error: {exc}")
        return f"{pid} {p['strategy_id']}: validation error ({exc})"

    if not result.ok_backtest or not result.ok_sharpe_bar:
        lifecycle.transition(conn, pid, "backtest_fail", detail=result.detail)
        return f"{pid} {p['strategy_id']}: failed backtest gate/Sharpe bar"
    lifecycle.transition(conn, pid, "backtest_pass", detail=result.detail)

    if not result.ok_holdout:
        lifecycle.transition(conn, pid, "holdout_fail", detail=result.detail)
        return f"{pid} {p['strategy_id']}: failed 12-month holdout"

    lifecycle.transition(conn, pid, "shadow", detail=result.detail)
    return (f"{pid} {p['strategy_id']}: entered shadow trading "
            f"(sel Sharpe {result.candidate_sharpe:.2f} > req {result.required_sharpe:.2f})")
