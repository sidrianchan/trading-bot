"""Strategy registry: versioned, immutable JSON records + an ACTIVE pointer.

Layout (git-committed, human-auditable):
    strategies/<strategy_id>/v<N>.json   one immutable record per version
    strategies/<strategy_id>/ACTIVE      the active version number

resolve_config() is the only integration point the live loops use. Its
contract: on ANY failure (missing dir, bad JSON, invalid params) it logs
a warning and returns the caller's fallback config — so a broken or
absent registry leaves the bots behaving exactly as they do today.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from evolve.families import get_family

# Anchored to the repo root so systemd services, cron jobs, and manual CLI
# invocations from any CWD all resolve the same registry.
STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "strategies"

REQUIRED_KEYS = {"strategy_id", "version", "family", "params", "status", "capital_fraction"}
VALID_STATUSES = {"active", "shadow", "retired"}


def _dir(strategy_id: str) -> Path:
    return STRATEGIES_DIR / strategy_id


def record_path(strategy_id: str, version: int) -> Path:
    return _dir(strategy_id) / f"v{version}.json"


def new_record(
    strategy_id: str,
    version: int,
    family: str,
    params: dict,
    *,
    status: str = "shadow",
    capital_fraction: float = 1.0,
    parent: str | None = None,
    proposal_id: str | None = None,
    created_by: str = "evolve",
    expected: dict | None = None,
    backtest_run_id: str | None = None,
) -> dict:
    return {
        "strategy_id": strategy_id,
        "version": version,
        "family": family,
        "params": params,
        "status": status,
        "capital_fraction": capital_fraction,
        "lineage": {"parent": parent, "proposal_id": proposal_id, "created_by": created_by},
        "provenance": {
            "backtest_run_id": backtest_run_id,
            "expected": expected,
            "shadow": None,
            "approved_by": None,
            "approved_at": None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_version(record: dict) -> Path:
    """Persist a version record. Versions are immutable: overwriting is refused."""
    missing = REQUIRED_KEYS - set(record)
    if missing:
        raise ValueError(f"registry record missing keys: {sorted(missing)}")
    if record["status"] not in VALID_STATUSES:
        raise ValueError(f"invalid status {record['status']!r}")
    # A record must always be buildable into a real config before it is stored.
    get_family(record["family"]).build_config(record["params"])

    path = record_path(record["strategy_id"], int(record["version"]))
    if path.exists():
        raise FileExistsError(f"registry version already exists (immutable): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    logger.info(f"Registry: wrote {path}")
    return path


def load_record(strategy_id: str, version: int) -> dict:
    return json.loads(record_path(strategy_id, version).read_text())


def active_version(strategy_id: str) -> int | None:
    pointer = _dir(strategy_id) / "ACTIVE"
    if not pointer.exists():
        return None
    try:
        return int(pointer.read_text().strip())
    except ValueError:
        logger.warning(f"Registry: corrupt ACTIVE pointer for {strategy_id}")
        return None


def set_active(strategy_id: str, version: int) -> None:
    if not record_path(strategy_id, version).exists():
        raise FileNotFoundError(f"cannot activate {strategy_id} v{version}: record not found")
    pointer = _dir(strategy_id) / "ACTIVE"
    pointer.write_text(f"{version}\n")
    logger.info(f"Registry: {strategy_id} ACTIVE -> v{version}")


def next_version(strategy_id: str) -> int:
    d = _dir(strategy_id)
    if not d.exists():
        return 1
    versions = [int(p.stem[1:]) for p in d.glob("v*.json")]
    return max(versions, default=0) + 1


def update_capital_fraction(strategy_id: str, version: int, fraction: float) -> None:
    """The one permitted mutation of a version record: the capital ramp.

    Params, lineage, and provenance stay immutable; capital_fraction is an
    operational sizing knob adjusted by ramp/demotion, and every change is
    visible in git history since strategies/ is committed.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"capital_fraction must be in (0, 1], got {fraction}")
    path = record_path(strategy_id, version)
    record = json.loads(path.read_text())
    record["capital_fraction"] = fraction
    path.write_text(json.dumps(record, indent=2) + "\n")
    logger.info(f"Registry: {strategy_id} v{version} capital_fraction -> {fraction:.0%}")


def load_active(strategy_id: str) -> dict | None:
    version = active_version(strategy_id)
    if version is None:
        return None
    return load_record(strategy_id, version)


def list_strategies() -> list[dict]:
    """Active record (plus version count) for every strategy in the registry."""
    if not STRATEGIES_DIR.exists():
        return []
    out = []
    for d in sorted(p for p in STRATEGIES_DIR.iterdir() if p.is_dir()):
        record = load_active(d.name)
        if record is None:
            continue
        record["n_versions"] = len(list(d.glob("v*.json")))
        out.append(record)
    return out


def resolve_config(strategy_id: str, fallback_cfg: Any, overrides: dict | None = None) -> Any:
    """Frozen config for the strategy's active version, or fallback_cfg on any failure."""
    try:
        record = load_active(strategy_id)
        if record is None:
            logger.debug(f"Registry: no active version for {strategy_id}; using fallback config")
            return fallback_cfg
        cfg = get_family(record["family"]).build_config(record["params"], overrides=overrides)
        logger.info(f"Registry: {strategy_id} resolved to v{record['version']}")
        return cfg
    except Exception as exc:
        logger.warning(f"Registry resolve failed for {strategy_id} ({exc}); using fallback config")
        return fallback_cfg


def capital_fraction(strategy_id: str) -> float:
    """Sizing multiplier for the active version; 1.0 on any failure (today's behavior)."""
    try:
        record = load_active(strategy_id)
        if record is None:
            return 1.0
        return float(record.get("capital_fraction", 1.0))
    except Exception as exc:
        logger.warning(f"Registry capital_fraction failed for {strategy_id} ({exc}); using 1.0")
        return 1.0


def seed_defaults(app_config: dict) -> list[str]:
    """Seed v1 records for the two production strategies (idempotent).

    Params mirror today's live configs exactly, so resolve_config() returns
    configs identical to the pre-registry behavior.
    """
    created: list[str] = []

    if active_version("etf_momentum") is None:
        family = get_family("dual_momentum_etf")
        write_version(new_record(
            "etf_momentum", 1, "dual_momentum_etf", family.default_params(),
            status="active", capital_fraction=1.0, created_by="migration",
            expected={"cagr": 0.249, "sharpe": 0.74, "max_dd": -0.678},
        ))
        set_active("etf_momentum", 1)
        created.append("etf_momentum/v1")

    if active_version("crypto_momentum") is None:
        crypto = app_config.get("crypto", {})
        abs_cfg = crypto.get("absolute_momentum", {})
        rel_cfg = crypto.get("relative_momentum", {})
        cb_cfg = crypto.get("circuit_breaker", {})
        params = {
            "abs_lookback": int(abs_cfg.get("lookback_days", 84)),
            "abs_skip": int(abs_cfg.get("skip_days", 14)),
            "rel_lookback": int(rel_cfg.get("lookback_days", 7)),
            "rel_skip": int(rel_cfg.get("skip_days", 14)),
            "cb_threshold": float(cb_cfg.get("max_drawdown_from_peak", -0.40)),
        }
        write_version(new_record(
            "crypto_momentum", 1, "crypto_momentum", params,
            status="active", capital_fraction=1.0, created_by="migration",
            expected={"cagr": 0.689, "sharpe": None, "max_dd": -0.552},
        ))
        set_active("crypto_momentum", 1)
        created.append("crypto_momentum/v1")

    return created
