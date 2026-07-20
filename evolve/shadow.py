"""Shadow paper-trading of candidate strategies.

Signal-level virtual portfolios — no broker orders, no interference with the
live loops. A daily job marks each shadow candidate's virtual book to market,
checks its drawdown envelope, and — once the minimum shadow period and
rebalance count are met — sends the human a promotion proposal via Telegram.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

import pandas as pd
from loguru import logger

from evolve import lifecycle
from evolve.families import get_family
from evolve.guardrails import HARD


@dataclass(frozen=True)
class ShadowRuntime:
    fetch_recent: Callable[[Any], pd.DataFrame]           # cfg -> prices
    compute: Callable[[pd.DataFrame, Any, float, Any], tuple]  # prices, state, value, cfg
    state_to_dict: Callable[[Any], dict]
    state_from_dict: Callable[[dict], Any]
    fresh_state: Callable[[float], Any]
    min_rebalances: int


def _etf_shadow() -> ShadowRuntime:
    import yfinance as yf

    from signals.dual_momentum import (
        V4State, compute_signal, state_from_dict, state_to_dict,
    )

    def fetch(cfg) -> pd.DataFrame:
        universe = list(cfg.risk_on) + list(cfg.risk_off_candidates) + [cfg.benchmark_filter]
        end = pd.Timestamp.now(tz="UTC").normalize()
        start = end - pd.Timedelta(days=550)
        raw = yf.download(sorted(set(universe)), start=start.date(), end=end.date(),
                          auto_adjust=True, progress=False, threads=False)
        return raw["Close"].dropna(how="all").ffill()

    return ShadowRuntime(
        fetch_recent=fetch,
        compute=compute_signal,
        state_to_dict=state_to_dict,
        state_from_dict=state_from_dict,
        fresh_state=lambda capital: V4State(peak=capital, cash_value=capital),
        min_rebalances=HARD.min_shadow_rebalances_etf,
    )


def _crypto_shadow() -> ShadowRuntime:
    import yfinance as yf

    from signals.crypto_momentum import (
        CryptoMomentumState, alpaca_to_yahoo_symbol, compute_crypto_signal,
        state_from_dict, state_to_dict,
    )

    def fetch(cfg) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=400)
        tickers = [alpaca_to_yahoo_symbol(s) for s in cfg.universe]
        raw = yf.download(tickers, start=start.date(), end=end.date(),
                          auto_adjust=True, progress=False, threads=False)
        prices = raw["Close"].rename(columns={"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"})
        return prices.dropna(how="all").ffill().dropna()

    return ShadowRuntime(
        fetch_recent=fetch,
        compute=compute_crypto_signal,
        state_to_dict=state_to_dict,
        state_from_dict=state_from_dict,
        fresh_state=lambda capital: CryptoMomentumState(peak=capital, cash_value=capital),
        min_rebalances=HARD.min_shadow_rebalances_crypto,
    )


SHADOW_RUNTIMES: dict[str, Callable[[], ShadowRuntime]] = {
    "dual_momentum_etf": _etf_shadow,
    "crypto_momentum": _crypto_shadow,
}


def _shadow_dir():
    d = lifecycle.EVOLVE_DIR / "shadow"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_book(proposal_id: str, capital: float, runtime: ShadowRuntime) -> dict:
    path = _shadow_dir() / f"{proposal_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "started": date.today().isoformat(),
        "cash": capital,
        "qty": 0.0,
        "holding": None,
        "peak": capital,
        "rebalances": 0,
        "signal_state": runtime.state_to_dict(runtime.fresh_state(capital)),
    }


def _save_book(proposal_id: str, book: dict) -> None:
    (_shadow_dir() / f"{proposal_id}.json").write_text(json.dumps(book, indent=2) + "\n")


def shadow_step(config: dict, *, prices_by_family: dict | None = None, today: date | None = None) -> None:
    """Daily idempotent step over every proposal in the 'shadow' state.

    prices_by_family / today are injectable for tests.
    """
    conn = lifecycle.connect()
    try:
        proposals = lifecycle.list_proposals(conn, state="shadow")
        if not proposals:
            logger.info("Evolve shadow-step: no candidates in shadow")
            return
        capital = float(config.get("evolve", {}).get("shadow_capital", 10_000.0))
        for p in proposals:
            try:
                _step_one(conn, p, capital, prices_by_family, today or date.today())
            except Exception as exc:
                logger.error(f"Shadow step failed for {p['proposal_id']} (will retry tomorrow): {exc}")
    finally:
        conn.close()


def _step_one(conn, p: dict, capital: float, prices_by_family: dict | None, today: date) -> None:
    family = get_family(p["family"])
    runtime = SHADOW_RUNTIMES[p["family"]]()
    cfg = family.build_config(p["params"])
    book = _load_book(p["proposal_id"], capital, runtime)

    if prices_by_family and p["family"] in prices_by_family:
        prices = prices_by_family[p["family"]]
    else:
        prices = runtime.fetch_recent(cfg)

    latest = prices.iloc[-1]
    value = book["cash"] + (book["qty"] * float(latest[book["holding"]]) if book["holding"] else 0.0)
    state = runtime.state_from_dict(book["signal_state"])
    signal, new_state = runtime.compute(prices, state, value, cfg)

    if signal.target != book["holding"]:
        # Virtual fill at the latest close
        book["cash"] = value
        book["qty"] = 0.0
        if signal.target:
            price = float(latest[signal.target])
            book["qty"] = book["cash"] / price
            book["cash"] = 0.0
        book["holding"] = signal.target
        book["rebalances"] += 1
        _journal(p, f"shadow rebalance -> {signal.target or 'CASH'}: {signal.decision_reason}")

    value = book["cash"] + (book["qty"] * float(latest[book["holding"]]) if book["holding"] else 0.0)
    book["peak"] = max(book["peak"], value)
    book["signal_state"] = runtime.state_to_dict(new_state)
    _save_book(p["proposal_id"], book)

    conn.execute(
        "INSERT OR IGNORE INTO shadow_marks (proposal_id, date, value, holding, detail)"
        " VALUES (?, ?, ?, ?, ?)",
        (p["proposal_id"], today.isoformat(), value, book["holding"] or "CASH",
         signal.decision_reason[:200]),
    )
    conn.commit()

    drawdown = value / book["peak"] - 1.0 if book["peak"] > 0 else 0.0
    envelope = _dd_envelope(p["expected"])
    if drawdown <= envelope:
        lifecycle.transition(
            conn, p["proposal_id"], "shadow_fail",
            detail=f"shadow drawdown {drawdown:.1%} breached envelope {envelope:.1%}",
        )
        _journal(p, f"shadow FAILED: drawdown {drawdown:.1%} <= envelope {envelope:.1%}")
        return

    days = (today - date.fromisoformat(book["started"])).days
    if days >= HARD.min_shadow_days and book["rebalances"] >= runtime.min_rebalances:
        lifecycle.transition(
            conn, p["proposal_id"], "pending_approval",
            detail=f"{days}d in shadow, {book['rebalances']} rebalances, value ${value:,.0f}",
        )
        _send_promotion_proposal(p, book, value, days)


def _dd_envelope(expected: dict) -> float:
    """Most-conservative applicable drawdown floor (negative fraction)."""
    exp_dd = expected.get("max_dd")
    floors = [-HARD.demote_dd_hard]
    if isinstance(exp_dd, (int, float)) and exp_dd < 0:
        floors.append(float(exp_dd) * HARD.demote_dd_multiplier)
    return max(floors)  # e.g. max(-0.30, -0.69) = -0.30


def _journal(p: dict, reason: str) -> None:
    from monitor.notify import journal_event

    journal_event({"event": "evolve_shadow", "bot": "Evolve",
                   "proposal_id": p["proposal_id"], "reason": reason})


def _send_promotion_proposal(p: dict, book: dict, value: float, days: int) -> None:
    from monitor.notify import TelegramNotifier

    TelegramNotifier().proposal(
        proposal_id=p["proposal_id"],
        strategy_id=p["strategy_id"],
        params=p["params"],
        hypothesis=p["hypothesis"],
        expected=p["expected"],
        shadow_summary=(
            f"{days} days in shadow since {book.get('started')}, "
            f"{book['rebalances']} rebalance(s), virtual book ${value:,.0f}"
        ),
    )
    logger.info(f"Evolve: promotion proposal sent for {p['proposal_id']}")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
