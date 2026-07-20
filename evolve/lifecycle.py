"""Proposal store and state machine for the evolution system.

SQLite at logs/evolve/evolve.db. Every state transition is row-logged with
timestamp, actor, and detail — the full audit trail of what the AI proposed
and what happened to it.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

EVOLVE_DIR = Path(__file__).resolve().parent.parent / "logs" / "evolve"

# proposal lifecycle; promotion to a live strategy requires a human "approved"
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "proposed": {"rejected_guardrail", "backtest_fail", "backtest_pass"},
    "backtest_pass": {"holdout_fail", "shadow"},
    "shadow": {"shadow_fail", "pending_approval"},
    "pending_approval": {"rejected", "approved"},
    "approved": {"active"},
    "active": {"demoted"},
    "demoted": {"retired"},
}
STATES = set(ALLOWED_TRANSITIONS) | {s for targets in ALLOWED_TRANSITIONS.values() for s in targets}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    cycle_id    TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    family      TEXT NOT NULL,
    params      TEXT NOT NULL,
    hypothesis  TEXT NOT NULL DEFAULT '',
    expected    TEXT NOT NULL DEFAULT '{}',
    state       TEXT NOT NULL,
    llm_call_id INTEGER,
    created_at  TEXT NOT NULL,
    decided_by  TEXT,
    decided_at  TEXT,
    note        TEXT
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT NOT NULL,
    from_state  TEXT NOT NULL,
    to_state    TEXT NOT NULL,
    actor       TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    model       TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    response    TEXT NOT NULL,
    usage       TEXT NOT NULL DEFAULT '{}',
    latency_ms  REAL
);
CREATE TABLE IF NOT EXISTS shadow_marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT NOT NULL,
    date        TEXT NOT NULL,
    value       REAL NOT NULL,
    holding     TEXT,
    detail      TEXT NOT NULL DEFAULT '',
    UNIQUE (proposal_id, date)
);
CREATE TABLE IF NOT EXISTS trials (
    family TEXT PRIMARY KEY,
    n      INTEGER NOT NULL DEFAULT 0
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return EVOLVE_DIR / "evolve.db"


def connect() -> sqlite3.Connection:
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def new_proposal_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"p-{stamp}-{uuid.uuid4().hex[:6]}"


def insert_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: str,
    cycle_id: str,
    strategy_id: str,
    family: str,
    params: dict,
    hypothesis: str,
    expected: dict,
    llm_call_id: int | None,
) -> None:
    conn.execute(
        "INSERT INTO proposals (proposal_id, cycle_id, strategy_id, family, params,"
        " hypothesis, expected, state, llm_call_id, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)",
        (proposal_id, cycle_id, strategy_id, family, json.dumps(params),
         hypothesis, json.dumps(expected), llm_call_id, _now()),
    )
    conn.execute(
        "INSERT INTO transitions (proposal_id, from_state, to_state, actor, detail, ts)"
        " VALUES (?, '', 'proposed', 'evolve', '', ?)",
        (proposal_id, _now()),
    )
    conn.commit()


def get_proposal(conn: sqlite3.Connection, proposal_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
    return _row_to_proposal(row) if row else None


def list_proposals(conn: sqlite3.Connection, state: str | None = None) -> list[dict]:
    if state:
        rows = conn.execute(
            "SELECT * FROM proposals WHERE state = ? ORDER BY created_at", (state,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM proposals ORDER BY created_at").fetchall()
    return [_row_to_proposal(r) for r in rows]


def _row_to_proposal(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d["params"])
    d["expected"] = json.loads(d["expected"])
    return d


def transition(
    conn: sqlite3.Connection,
    proposal_id: str,
    to_state: str,
    *,
    actor: str = "evolve",
    detail: str = "",
) -> None:
    """Move a proposal to a new state; invalid transitions raise ValueError."""
    proposal = get_proposal(conn, proposal_id)
    if proposal is None:
        raise ValueError(f"unknown proposal: {proposal_id}")
    from_state = proposal["state"]
    if to_state not in ALLOWED_TRANSITIONS.get(from_state, set()):
        raise ValueError(f"invalid transition {from_state} -> {to_state} for {proposal_id}")

    decided = to_state in {"approved", "rejected"}
    conn.execute(
        "UPDATE proposals SET state = ?, decided_by = COALESCE(?, decided_by),"
        " decided_at = COALESCE(?, decided_at), note = COALESCE(NULLIF(?, ''), note)"
        " WHERE proposal_id = ?",
        (to_state, actor if decided else None, _now() if decided else None, detail, proposal_id),
    )
    conn.execute(
        "INSERT INTO transitions (proposal_id, from_state, to_state, actor, detail, ts)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (proposal_id, from_state, to_state, actor, detail, _now()),
    )
    conn.commit()
    logger.info(f"Evolve: {proposal_id} {from_state} -> {to_state} ({actor}) {detail}")


def log_llm_call(
    conn: sqlite3.Connection,
    *,
    model: str,
    prompt: str,
    response: str,
    usage: dict,
    latency_ms: float | None = None,
) -> int:
    """Persist an LLM interaction; committed BEFORE any proposal is acted on."""
    cur = conn.execute(
        "INSERT INTO llm_calls (ts, model, prompt, response, usage, latency_ms)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (_now(), model, prompt, response, json.dumps(usage), latency_ms),
    )
    conn.commit()
    return int(cur.lastrowid)


def bump_trials(conn: sqlite3.Connection, family: str, k: int = 1) -> None:
    conn.execute(
        "INSERT INTO trials (family, n) VALUES (?, ?)"
        " ON CONFLICT(family) DO UPDATE SET n = n + excluded.n",
        (family, k),
    )
    conn.commit()


def trial_count(conn: sqlite3.Connection, family: str) -> int:
    row = conn.execute("SELECT n FROM trials WHERE family = ?", (family,)).fetchone()
    return int(row["n"]) if row else 0


def count_in_states(conn: sqlite3.Connection, states: set[str]) -> int:
    marks = ",".join("?" * len(states))
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM proposals WHERE state IN ({marks})", tuple(states)
    ).fetchone()
    return int(row["n"])
