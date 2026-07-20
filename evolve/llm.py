"""Anthropic API layer for the evolution agent.

One structured call per weekly cycle: the digest goes in, a schema-validated
ProposalBatch comes out. Every interaction (full prompt, raw response, usage)
is persisted to logs/evolve/llm_log.jsonl AND the evolve.db llm_calls table
BEFORE any proposal is acted on. An API failure aborts the cycle cleanly.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from loguru import logger

from evolve import lifecycle
from evolve.families import FAMILIES
from evolve.guardrails import HARD

DEFAULT_MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000


class ProposalParseError(RuntimeError):
    """The LLM reply could not be parsed into a valid proposal batch."""


STRATEGY_FAMILY = {"etf_momentum": "dual_momentum_etf", "crypto_momentum": "crypto_momentum"}


def _params_schema(family_id: str) -> dict:
    family = FAMILIES[family_id]
    kind_map = {"int": "integer", "float": "number"}
    return {
        "type": "object",
        "properties": {p.name: {"type": kind_map[p.kind]} for p in family.params},
        "required": [p.name for p in family.params],
        "additionalProperties": False,
    }


def proposal_batch_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "analysis_summary": {
                "type": "string",
                "description": "Short assessment of current performance and regime.",
            },
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "strategy_id": {"type": "string", "enum": sorted(STRATEGY_FAMILY)},
                        "family": {"type": "string", "enum": sorted(FAMILIES)},
                        "params": {"anyOf": [_params_schema(f) for f in sorted(FAMILIES)]},
                        "hypothesis": {"type": "string"},
                        "expected": {
                            "type": "object",
                            "properties": {
                                "cagr": {"type": "number"},
                                "sharpe": {"type": "number"},
                                "max_dd": {"type": "number"},
                            },
                            "required": ["cagr", "sharpe", "max_dd"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["strategy_id", "family", "params", "hypothesis", "expected"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["analysis_summary", "proposals"],
        "additionalProperties": False,
    }


def system_prompt() -> str:
    """Static system prompt: role, rules, and the bounded search space."""
    family_lines = []
    for family in FAMILIES.values():
        family_lines.append(f"- {family.family_id}: {family.description}")
        for spec in family.params:
            family_lines.append(
                f"    {spec.name} ({spec.kind}): [{spec.lo}, {spec.hi}] step {spec.step}"
            )
    families_text = "\n".join(family_lines)
    return f"""You are the strategy-evolution analyst for a small systematic trading system \
(paper trading on Alpaca: a leveraged-ETF dual-momentum bot and a BTC/ETH momentum bot).

Your job each week: review the evidence digest and decide whether any strategy parameter \
changes are worth TESTING. You propose candidates; you do not deploy anything. Every \
candidate must pass backtest gates, a 12-month out-of-sample holdout, and >= \
{HARD.min_shadow_days} days of shadow paper trading, and then a HUMAN must approve it.

Strategy families and bounded parameter grids (values outside the grid are rejected):
{families_text}

Rules:
- Propose at most {HARD.max_candidates_per_cycle} candidates. PREFER ZERO when evidence is \
weak — most weeks the right answer is no change. Overfitting is the main failure mode.
- Never re-propose a parameter set listed as rejected or failed in the digest.
- Never propose loosening a circuit breaker as a way to improve returns.
- A candidate must have a specific, falsifiable hypothesis grounded in the digest \
(e.g. a regime observation), not generic parameter twiddling.
- expected.max_dd is a negative fraction (e.g. -0.55); expected.cagr and expected.sharpe \
are your honest estimates — they become the demotion envelope if promoted.
- Context: this system previously tried and abandoned intraday TA, ORB/gap, VWAP \
mean-reversion, and monthly factor-model strategies — all failed backtest gates. \
Do not steer back toward those.

Respond with the required JSON only."""


def propose(
    digest: str,
    config: dict,
    conn,
    client=None,
) -> tuple[dict, int]:
    """Run the proposal call. Returns (batch, llm_call_id).

    The raw exchange is persisted before parsing, so even a malformed reply
    leaves a complete audit record.
    """
    model = config.get("evolve", {}).get("model", DEFAULT_MODEL)
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    system = system_prompt()
    started = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=system,
        output_config={"format": {"type": "json_schema", "schema": proposal_batch_schema()}},
        messages=[{"role": "user", "content": digest}],
    )
    latency_ms = (time.monotonic() - started) * 1000
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = _usage_dict(response)

    call_id = lifecycle.log_llm_call(
        conn, model=model, prompt=f"[system]\n{system}\n\n[user]\n{digest}",
        response=text, usage=usage, latency_ms=latency_ms,
    )
    _append_jsonl_audit(model, digest, text, usage, latency_ms)
    logger.info(f"Evolve LLM call #{call_id}: {usage} in {latency_ms:.0f}ms")

    batch = parse_batch(text)
    return batch, call_id


def parse_batch(text: str) -> dict:
    """Parse + structurally validate the reply. Family-level param validation
    (bounds, grids, hard limits) happens later in the cycle, per proposal."""
    try:
        batch = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProposalParseError(f"LLM reply is not valid JSON: {exc}") from exc
    if not isinstance(batch, dict) or "proposals" not in batch:
        raise ProposalParseError("LLM reply missing 'proposals'")
    proposals = batch["proposals"]
    if not isinstance(proposals, list):
        raise ProposalParseError("'proposals' is not a list")
    for i, p in enumerate(proposals):
        missing = {"strategy_id", "family", "params", "hypothesis", "expected"} - set(p)
        if missing:
            raise ProposalParseError(f"proposal {i} missing keys: {sorted(missing)}")
        if p["family"] not in FAMILIES:
            raise ProposalParseError(f"proposal {i} unknown family {p['family']!r}")
        if STRATEGY_FAMILY.get(p["strategy_id"]) != p["family"]:
            raise ProposalParseError(
                f"proposal {i}: strategy {p['strategy_id']!r} does not belong to family {p['family']!r}"
            )
        expected = p["expected"]
        if not all(isinstance(expected.get(k), (int, float)) for k in ("cagr", "sharpe", "max_dd")):
            raise ProposalParseError(f"proposal {i} expected metrics malformed: {expected}")
    return batch


def _usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        k: getattr(usage, k, None)
        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens")
    }


def _append_jsonl_audit(model: str, digest: str, text: str, usage: dict, latency_ms: float) -> None:
    lifecycle.EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "digest": digest,
        "response": text,
        "usage": usage,
        "latency_ms": latency_ms,
    }
    with open(lifecycle.EVOLVE_DIR / "llm_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
