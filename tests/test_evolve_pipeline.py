"""Tests for the evolution pipeline: lifecycle DB, LLM layer, validator (Phase 2)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evolve import lifecycle, llm
from evolve.families import get_family
from evolve.validator import equity_metrics, required_sharpe, validate_candidate


@pytest.fixture(autouse=True)
def isolated_evolve_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "EVOLVE_DIR", tmp_path / "evolve")


@pytest.fixture
def conn():
    c = lifecycle.connect()
    yield c
    c.close()


def _insert(conn, pid="p-test-1", params=None):
    lifecycle.insert_proposal(
        conn, proposal_id=pid, cycle_id="c-test", strategy_id="etf_momentum",
        family="dual_momentum_etf",
        params=params or get_family("dual_momentum_etf").default_params(),
        hypothesis="test", expected={"cagr": 0.2, "sharpe": 0.7, "max_dd": -0.5},
        llm_call_id=None,
    )


# ── Lifecycle ───────────────────────────────────────────────────────────


def test_proposal_round_trip_and_valid_transitions(conn):
    _insert(conn)
    p = lifecycle.get_proposal(conn, "p-test-1")
    assert p["state"] == "proposed"

    lifecycle.transition(conn, "p-test-1", "backtest_pass", detail="gates pass")
    lifecycle.transition(conn, "p-test-1", "shadow")
    lifecycle.transition(conn, "p-test-1", "pending_approval")
    lifecycle.transition(conn, "p-test-1", "approved", actor="sid")
    p = lifecycle.get_proposal(conn, "p-test-1")
    assert p["state"] == "approved"
    assert p["decided_by"] == "sid"
    assert p["decided_at"] is not None

    rows = conn.execute("SELECT COUNT(*) AS n FROM transitions WHERE proposal_id='p-test-1'").fetchone()
    assert rows["n"] == 5  # insert + 4 transitions


def test_invalid_transition_rejected(conn):
    _insert(conn)
    with pytest.raises(ValueError, match="invalid transition"):
        lifecycle.transition(conn, "p-test-1", "approved")  # proposed -> approved skips gates
    with pytest.raises(ValueError, match="unknown proposal"):
        lifecycle.transition(conn, "p-nope", "shadow")


def test_terminal_states_cannot_move(conn):
    _insert(conn)
    lifecycle.transition(conn, "p-test-1", "rejected_guardrail", detail="cb too loose")
    with pytest.raises(ValueError, match="invalid transition"):
        lifecycle.transition(conn, "p-test-1", "backtest_pass")


def test_trials_counter(conn):
    assert lifecycle.trial_count(conn, "dual_momentum_etf") == 0
    lifecycle.bump_trials(conn, "dual_momentum_etf")
    lifecycle.bump_trials(conn, "dual_momentum_etf")
    assert lifecycle.trial_count(conn, "dual_momentum_etf") == 2


# ── LLM layer ───────────────────────────────────────────────────────────


def _fake_client(reply_text: str):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=reply_text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        return response

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return client, calls


VALID_BATCH = {
    "analysis_summary": "Momentum regime intact; test a faster ETF lookback.",
    "proposals": [{
        "strategy_id": "etf_momentum",
        "family": "dual_momentum_etf",
        "params": {"abs_lookback": 84, "rel_lookback": 63, "skip": 21,
                   "cb_threshold": 0.25, "reentry_confirmation_months": 2},
        "hypothesis": "Faster abs filter reacts to V-shaped recoveries",
        "expected": {"cagr": 0.22, "sharpe": 0.7, "max_dd": -0.6},
    }],
}


def test_propose_valid_reply_returns_batch_and_audits(conn):
    client, calls = _fake_client(json.dumps(VALID_BATCH))
    batch, call_id = llm.propose("digest text", {"evolve": {"model": "claude-opus-4-8"}}, conn, client=client)
    assert len(batch["proposals"]) == 1
    assert calls["model"] == "claude-opus-4-8"
    assert calls["output_config"]["format"]["type"] == "json_schema"

    row = conn.execute("SELECT * FROM llm_calls WHERE id = ?", (call_id,)).fetchone()
    assert row is not None
    assert "digest text" in row["prompt"]
    assert json.loads(row["usage"])["input_tokens"] == 100
    assert (lifecycle.EVOLVE_DIR / "llm_log.jsonl").exists()


def test_propose_malformed_reply_raises_but_still_audits(conn):
    client, _ = _fake_client("{not json")
    with pytest.raises(llm.ProposalParseError):
        llm.propose("digest", {}, conn, client=client)
    n = conn.execute("SELECT COUNT(*) AS n FROM llm_calls").fetchone()["n"]
    assert n == 1  # audit row written before parsing failed


def test_parse_batch_rejects_wrong_family_pairing():
    bad = json.loads(json.dumps(VALID_BATCH))
    bad["proposals"][0]["strategy_id"] = "crypto_momentum"
    with pytest.raises(llm.ProposalParseError, match="does not belong"):
        llm.parse_batch(json.dumps(bad))


def test_schema_is_structured_output_compatible():
    schema = llm.proposal_batch_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["proposals"]["items"]["additionalProperties"] is False


# ── Validator ───────────────────────────────────────────────────────────


def _synthetic_etf_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-02", "2026-07-01")
    n = len(dates)
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, n)))
    data = {
        "SPY": spy,
        "TQQQ": 100 * np.exp(np.cumsum(rng.normal(0.0010, 0.030, n))),
        "UPRO": 100 * np.exp(np.cumsum(rng.normal(0.0010, 0.028, n))),
        "SOXL": 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.035, n))),
        "TLT": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, n))),
    }
    return pd.DataFrame(data, index=dates)


def test_required_sharpe_rises_with_trials():
    base = required_sharpe(0.70, 0)
    assert base == pytest.approx(0.80)
    assert required_sharpe(0.70, 10) > base
    assert required_sharpe(0.70, 100) > required_sharpe(0.70, 10)


def test_equity_metrics_shape():
    equity = pd.Series(
        np.linspace(100, 150, 300), index=pd.bdate_range("2024-01-01", periods=300)
    )
    m = equity_metrics(equity, 252)
    assert m["cagr"] > 0
    assert m["max_drawdown"] <= 0
    assert "sharpe" in m


def test_validate_candidate_on_synthetic_prices():
    family = get_family("dual_momentum_etf")
    params = family.default_params()
    result = validate_candidate(
        "dual_momentum_etf", params, params, n_trials=0, prices=_synthetic_etf_prices()
    )
    # Structural assertions — synthetic data need not pass the gates
    assert isinstance(result.ok_backtest, bool)
    assert len(result.gates) == 4
    assert result.selection_metrics and result.holdout_metrics
    # candidate == incumbent, so the rising bar must fail it (bar = own sharpe + 0.10)
    assert result.candidate_sharpe == pytest.approx(result.incumbent_sharpe)
    assert not result.ok_sharpe_bar
    # holdout window is the trailing 12 months of the full run
    assert result.detail


def test_validate_candidate_holdout_split():
    prices = _synthetic_etf_prices()
    family = get_family("dual_momentum_etf")
    params = {**family.default_params(), "abs_lookback": 84}
    result = validate_candidate(
        "dual_momentum_etf", params, family.default_params(), n_trials=3, prices=prices
    )
    assert result.required_sharpe > result.incumbent_sharpe + 0.10
    assert result.holdout_metrics  # computed on trailing-12-month slice
