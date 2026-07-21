"""Tests for the strategy registry, families, and guardrails (evolve Phase 1)."""
from __future__ import annotations

import json

import pytest

from evolve import registry
from evolve.families import FAMILIES, ParamValidationError, get_family
from evolve.guardrails import HARD, check_hard_limits
from signals.crypto_momentum import CryptoMomentumConfig
from signals.dual_momentum import V4Config

APP_CONFIG = {
    "crypto": {
        "capital": 3_000,
        "universe": ["BTC/USD", "ETH/USD"],
        "stable": "USDC/USD",
        "absolute_momentum": {"lookback_days": 84, "skip_days": 14},
        "relative_momentum": {"lookback_days": 7, "skip_days": 14},
        "circuit_breaker": {"max_drawdown_from_peak": -0.40},
    }
}


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "STRATEGIES_DIR", tmp_path / "strategies")


# ── Families ────────────────────────────────────────────────────────────


def test_etf_default_params_build_production_config():
    family = get_family("dual_momentum_etf")
    assert family.build_config(family.default_params()) == V4Config()


def test_crypto_default_params_build_production_config():
    family = get_family("crypto_momentum")
    cfg = family.build_config(family.default_params(), overrides={"capital": 3_000.0})
    assert cfg == CryptoMomentumConfig()


def test_out_of_bounds_param_rejected():
    family = get_family("dual_momentum_etf")
    params = {**family.default_params(), "abs_lookback": 300}
    with pytest.raises(ParamValidationError, match="outside bounds"):
        family.build_config(params)


def test_off_grid_param_rejected():
    family = get_family("dual_momentum_etf")
    params = {**family.default_params(), "abs_lookback": 100}  # not on 63 + 21k grid
    with pytest.raises(ParamValidationError, match="step grid"):
        family.build_config(params)


def test_unknown_and_missing_params_rejected():
    family = get_family("crypto_momentum")
    with pytest.raises(ParamValidationError, match="unknown params"):
        family.build_config({**family.default_params(), "leverage": 10})
    with pytest.raises(ParamValidationError, match="missing params"):
        family.build_config({"abs_lookback": 84})


def test_wrong_type_rejected():
    family = get_family("dual_momentum_etf")
    params = {**family.default_params(), "skip": "21"}
    with pytest.raises(ParamValidationError, match="not an int"):
        family.build_config(params)


def test_unknown_family_rejected():
    with pytest.raises(ParamValidationError, match="unknown strategy family"):
        get_family("llm_written_python")


# ── Guardrails ──────────────────────────────────────────────────────────


def test_etf_cb_threshold_hard_limit():
    assert check_hard_limits("dual_momentum_etf", {"cb_threshold": 0.45})
    assert not check_hard_limits("dual_momentum_etf", {"cb_threshold": 0.25})
    # ParamSpec bounds (0.35 max) are tighter than the hard floor (0.40) — both reject 0.45
    family = get_family("dual_momentum_etf")
    with pytest.raises(ParamValidationError):
        family.build_config({**family.default_params(), "cb_threshold": 0.45})


def test_crypto_cb_threshold_hard_limit():
    assert check_hard_limits("crypto_momentum", {"cb_threshold": -0.60})
    assert not check_hard_limits("crypto_momentum", {"cb_threshold": -0.40})


def test_hard_limits_are_frozen():
    with pytest.raises(Exception):
        HARD.max_candidates_per_cycle = 100  # type: ignore[misc]


# ── Registry ────────────────────────────────────────────────────────────


def test_write_and_load_round_trip():
    family = get_family("dual_momentum_etf")
    record = registry.new_record(
        "etf_momentum", 1, "dual_momentum_etf", family.default_params(), status="active"
    )
    registry.write_version(record)
    registry.set_active("etf_momentum", 1)
    loaded = registry.load_active("etf_momentum")
    assert loaded["params"] == family.default_params()
    assert loaded["version"] == 1
    assert loaded["capital_fraction"] == 1.0


def test_versions_are_immutable():
    family = get_family("dual_momentum_etf")
    record = registry.new_record(
        "etf_momentum", 1, "dual_momentum_etf", family.default_params(), status="active"
    )
    registry.write_version(record)
    with pytest.raises(FileExistsError):
        registry.write_version(record)


def test_write_version_rejects_invalid_params():
    record = registry.new_record(
        "etf_momentum", 1, "dual_momentum_etf", {"abs_lookback": 300}, status="active"
    )
    with pytest.raises(ParamValidationError):
        registry.write_version(record)


def test_set_active_requires_existing_record():
    with pytest.raises(FileNotFoundError):
        registry.set_active("etf_momentum", 7)


# ── resolve_config: the no-breakage guarantee ───────────────────────────


def test_resolve_config_missing_registry_returns_fallback():
    fallback = V4Config()
    assert registry.resolve_config("etf_momentum", fallback) is fallback
    assert registry.capital_fraction("etf_momentum") == 1.0


def test_resolve_config_corrupt_record_returns_fallback():
    registry.seed_defaults(APP_CONFIG)
    path = registry.record_path("etf_momentum", 1)
    path.write_text("{not json")
    fallback = V4Config(abs_lookback=999)
    assert registry.resolve_config("etf_momentum", fallback) is fallback


def test_seeded_registry_reproduces_todays_configs():
    created = registry.seed_defaults(APP_CONFIG)
    assert set(created) == {"etf_momentum/v1", "crypto_momentum/v1"}

    etf_cfg = registry.resolve_config("etf_momentum", fallback_cfg=None)
    assert etf_cfg == V4Config()

    crypto_cfg = registry.resolve_config(
        "crypto_momentum", fallback_cfg=None, overrides={"capital": 3_000.0}
    )
    assert crypto_cfg == CryptoMomentumConfig()

    # Seeding is idempotent
    assert registry.seed_defaults(APP_CONFIG) == []


def test_seeded_records_are_valid_json_with_provenance():
    registry.seed_defaults(APP_CONFIG)
    record = json.loads(registry.record_path("etf_momentum", 1).read_text())
    assert record["status"] == "active"
    assert record["lineage"]["created_by"] == "migration"
    assert record["provenance"]["expected"]["max_dd"] == -0.678


def test_list_strategies_shows_seeded():
    registry.seed_defaults(APP_CONFIG)
    listed = registry.list_strategies()
    assert {r["strategy_id"] for r in listed} == {"crypto_momentum", "etf_momentum"}
    assert all(r["n_versions"] == 1 for r in listed)


def test_all_families_default_params_are_on_grid():
    for family in FAMILIES.values():
        assert family.validate_params(family.default_params()) == []


# ── Live-loop wiring ────────────────────────────────────────────────────
#
# The registry only affects real trading through these two call sites. If the
# wiring regresses, promotion silently becomes a no-op: `evolve approve` would
# write a new version and flip ACTIVE while the bots kept running v1.


def test_etf_loop_resolves_promoted_version():
    """The ETF loop's config follows the ACTIVE pointer, not config.yaml."""
    registry.seed_defaults(APP_CONFIG)
    family = get_family("dual_momentum_etf")
    params = {**family.default_params(), "abs_lookback": 105}
    registry.write_version(registry.new_record(
        "etf_momentum", 2, "dual_momentum_etf", params,
        status="active", capital_fraction=HARD.new_strategy_capital_fraction,
        parent="etf_momentum/v1",
    ))
    registry.set_active("etf_momentum", 2)

    cfg = registry.resolve_config("etf_momentum", V4Config())
    assert cfg.abs_lookback == 105
    assert registry.capital_fraction("etf_momentum") == HARD.new_strategy_capital_fraction


def test_crypto_loop_resolves_promoted_version_and_keeps_capital_override():
    """Crypto capital is a runtime override, never an evolvable param."""
    registry.seed_defaults(APP_CONFIG)
    family = get_family("crypto_momentum")
    params = {**family.default_params(), "rel_lookback": 14}
    registry.write_version(registry.new_record(
        "crypto_momentum", 2, "crypto_momentum", params,
        status="active", parent="crypto_momentum/v1",
    ))
    registry.set_active("crypto_momentum", 2)

    from agent.crypto_loop import CryptoPaperLoop

    cfg = CryptoPaperLoop._build_config(APP_CONFIG["crypto"])
    assert cfg.rel_lookback == 14
    assert cfg.capital == 3_000.0


def test_live_loops_fall_back_to_yaml_when_registry_empty():
    """An absent registry must leave both loops behaving exactly as before."""
    from agent.crypto_loop import CryptoPaperLoop

    assert registry.resolve_config("etf_momentum", V4Config()) == V4Config()
    assert registry.capital_fraction("etf_momentum") == 1.0
    assert CryptoPaperLoop._build_config(APP_CONFIG["crypto"]) == CryptoMomentumConfig()


def test_corrupt_registry_record_falls_back_instead_of_raising():
    """A bad record must not take the live loop down."""
    registry.seed_defaults(APP_CONFIG)
    registry.record_path("etf_momentum", 1).write_text("{ not valid json")
    assert registry.resolve_config("etf_momentum", V4Config()) == V4Config()
    assert registry.capital_fraction("etf_momentum") == 1.0
