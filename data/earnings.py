"""Earnings-date lookup with Alpaca → yfinance fallback.

Used by the swing-trade earnings filter (skip entries within N days of the
next earnings release).

Phase A scaffolding — implementation lands in Phase C.
"""
from __future__ import annotations

from datetime import date


def next_earnings_date(ticker: str) -> date | None:
    """Return the next scheduled earnings date for ``ticker``.

    Tries Alpaca's corporate-actions endpoint first, falls back to
    ``yfinance.Ticker(ticker).get_calendar()``. Cached for 24 hours via
    ``data.cache.Cache``. Returns ``None`` when no upcoming date is known.

    Phase A: stubbed — returns ``None`` so swing setups are never blocked
    during the initial backtest scaffolding. Real implementation lands in
    Phase C.
    """
    return None


def is_within_blackout(ticker: str, today: date, blackout_days: int) -> bool:
    """True if ``ticker``'s next earnings date is within ``blackout_days``."""
    nxt = next_earnings_date(ticker)
    if nxt is None:
        return False
    return 0 <= (nxt - today).days <= blackout_days
