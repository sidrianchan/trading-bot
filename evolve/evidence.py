"""Performance-evidence digest for the evolution LLM.

Builds a compact markdown digest (never raw price ticks): recent journal
activity, bot states, a market-regime snapshot, the strategy registry, and
open/past proposals. Every section degrades gracefully — a data source
being unavailable yields an "(unavailable)" line, never an exception.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from evolve import lifecycle, registry
from evolve.families import FAMILIES
from evolve.guardrails import HARD


def build_digest(config: dict, weeks: int | None = None) -> str:
    weeks = weeks or int(config.get("evolve", {}).get("digest_weeks", 8))
    sections = [
        f"# Trading performance digest — {datetime.now(timezone.utc).date().isoformat()}",
        _section("Strategy registry", _registry_section),
        _section("Bot state", _bot_state_section),
        _section(f"Trade journal (last {weeks} weeks)", lambda: _journal_section(weeks)),
        _section("Market regime snapshot", _regime_section),
        _section("Evolution history", _history_section),
        _section("Constraints (immutable)", _constraints_section),
    ]
    return "\n\n".join(sections)


def _section(title: str, builder) -> str:
    try:
        body = builder()
    except Exception as exc:
        logger.warning(f"Digest section '{title}' failed: {exc}")
        body = "(unavailable)"
    return f"## {title}\n{body}"


def _registry_section() -> str:
    strategies = registry.list_strategies()
    if not strategies:
        return "(registry empty)"
    lines = []
    for rec in strategies:
        expected = (rec.get("provenance") or {}).get("expected") or {}
        lines.append(
            f"- **{rec['strategy_id']}** v{rec['version']} [{rec['status']}] "
            f"family={rec['family']} capital_fraction={rec['capital_fraction']:.0%}"
        )
        lines.append(f"  params: {json.dumps(rec['params'], sort_keys=True)}")
        if expected:
            lines.append(f"  backtest expectations: {json.dumps(expected, sort_keys=True)}")
    return "\n".join(lines)


def _bot_state_section() -> str:
    lines = []
    for label, path in [("ETF", Path("logs/momentum_state.json")),
                        ("Crypto", Path("logs/crypto_state.json"))]:
        if not path.exists():
            lines.append(f"- {label}: no state file")
            continue
        state = json.loads(path.read_text())
        parts = [f"holding={state.get('last_target') or 'CASH'}",
                 f"peak=${state.get('peak', 0):,.0f}"]
        if state.get("in_cb"):
            parts.append(f"IN CIRCUIT BREAKER (confirm {state.get('cb_confirm_count', 0)})")
        if state.get("last_eval_date"):
            parts.append(f"last_eval={state['last_eval_date']}")
        lines.append(f"- {label}: " + ", ".join(parts))
    return "\n".join(lines)


def _journal_section(weeks: int) -> str:
    from monitor.notify import _journal_path

    path = _journal_path()
    if not path.exists():
        return "(no journal)"
    cutoff = (datetime.now(timezone.utc) - timedelta(weeks=weeks)).date().isoformat()
    counts: dict[tuple[str, str], int] = {}
    recent: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = str(entry.get("ts", ""))
            if ts[:10] < cutoff:
                continue
            key = (str(entry.get("bot", "?")), str(entry.get("event", "?")))
            counts[key] = counts.get(key, 0) + 1
            desc = f"{ts[:10]} [{entry.get('bot')}] {entry.get('event')}"
            if entry.get("event") == "trade":
                desc += f" {entry.get('action')} {entry.get('symbol')} ${entry.get('value', 0):,.0f}"
            elif entry.get("event") == "rebalance":
                desc += f" -> {entry.get('target')}"
            elif entry.get("event") == "circuit_breaker":
                desc += f" at {entry.get('drawdown', 0):.1%}"
            if entry.get("reason"):
                desc += f" — {entry['reason']}"
            recent.append(desc)
    if not recent:
        return "(no events in window)"
    lines = ["Event counts: " + ", ".join(f"{bot}/{event}={n}" for (bot, event), n in sorted(counts.items()))]
    lines += [f"- {d}" for d in recent[-30:]]
    return "\n".join(lines)


def _regime_section() -> str:
    import pandas as pd
    import yfinance as yf

    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=400)
    raw = yf.download(["SPY", "BTC-USD"], start=start.date(), end=end.date(),
                      auto_adjust=True, progress=False, threads=False)
    closes = raw["Close"].ffill()
    lines = []

    spy = closes["SPY"].dropna()
    if len(spy) > 200:
        ma200 = spy.rolling(200).mean().iloc[-1]
        vol20 = spy.pct_change().tail(20).std() * (252 ** 0.5)
        lines.append(
            f"- SPY: {spy.iloc[-1]:,.0f} vs 200dma {ma200:,.0f} "
            f"({'above' if spy.iloc[-1] > ma200 else 'BELOW'}), 20d realized vol {vol20:.0%}"
        )
    btc = closes["BTC-USD"].dropna()
    if len(btc) > 90:
        ret90 = btc.iloc[-1] / btc.iloc[-90] - 1.0
        dd = btc.iloc[-1] / btc.tail(90).max() - 1.0
        lines.append(f"- BTC: 90d return {ret90:+.1%}, drawdown from 90d high {dd:+.1%}")
    return "\n".join(lines) or "(no data)"


def _history_section() -> str:
    if not lifecycle.db_path().exists():
        return "(no evolution history yet)"
    conn = lifecycle.connect()
    try:
        proposals = lifecycle.list_proposals(conn)
        if not proposals:
            return "(no proposals yet)"
        lines = []
        for p in proposals[-25:]:
            lines.append(
                f"- {p['proposal_id']} {p['strategy_id']} state={p['state']} "
                f"params={json.dumps(p['params'], sort_keys=True)}"
                + (f" note={p['note']}" if p.get("note") else "")
            )
        lines.append("Do NOT re-propose parameter sets that were previously rejected or failed.")
        return "\n".join(lines)
    finally:
        conn.close()


def _constraints_section() -> str:
    lines = ["Strategy families and their bounded parameter grids "
             "(proposals outside these grids are rejected, never clamped):"]
    for family in FAMILIES.values():
        lines.append(f"- **{family.family_id}**: {family.description}")
        for spec in family.params:
            lines.append(f"    {spec.name}: {spec.kind} in [{spec.lo}, {spec.hi}] step {spec.step}")
        lines.append(f"    fixed (not evolvable): {json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in family.fixed.items()})}")
    lines.append(
        f"Hard limits: max {HARD.max_candidates_per_cycle} candidates per cycle; "
        f"max {HARD.max_concurrent_shadows} concurrent shadows; shadow >= {HARD.min_shadow_days} days; "
        f"new strategies capped at {HARD.new_strategy_capital_fraction:.0%} capital for {HARD.ramp_days} days; "
        f"auto-demotion at {HARD.demote_dd_multiplier}x expected max drawdown. "
        f"Risk limits and kill switch are outside your control."
    )
    return "\n".join(lines)
