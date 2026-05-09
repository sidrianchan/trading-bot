"""Earnings-date lookup with Alpaca → yfinance fallback.

Used by the swing-trade earnings filter (skip entries within N days of the
next earnings release). Cached for 24h via ``data.cache.Cache``.
"""
from __future__ import annotations

import os
from datetime import date, datetime

import pandas as pd
from loguru import logger

from data.cache import Cache

_cache = Cache(ttl_hours=24)


def _cache_key(ticker: str) -> str:
    return f"earnings_next_{ticker.upper()}"


def next_earnings_date(ticker: str) -> date | None:
    """Return the next scheduled earnings date for ``ticker`` or ``None``.

    Tries Alpaca's corporate-actions API first, falls back to yfinance.
    Cached for 24 hours.
    """
    key = _cache_key(ticker)
    cached = _cache.get(key)
    if cached is not None:
        try:
            return cached.iloc[0, 0] if hasattr(cached, "iloc") else cached
        except Exception:
            pass

    dt = _alpaca_next_earnings(ticker) or _yfinance_next_earnings(ticker)
    if dt is not None:
        try:
            _cache.set(key, pd.DataFrame([[dt]]))
        except Exception:
            pass
    return dt


def is_within_blackout(ticker: str, today: date, blackout_days: int) -> bool:
    """True if ``ticker``'s next earnings date is within ``blackout_days``."""
    nxt = next_earnings_date(ticker)
    if nxt is None:
        return False
    return 0 <= (nxt - today).days <= blackout_days


# ──────────────────────────────────────────────────────────────────────────


def _alpaca_next_earnings(ticker: str) -> date | None:
    try:
        from alpaca.data.historical.corporate_actions import CorporateActionsClient  # type: ignore
        from alpaca.data.requests import CorporateActionsRequest                      # type: ignore
    except Exception:
        return None

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return None

    try:
        client = CorporateActionsClient(api_key=api_key, secret_key=secret_key)
        today = date.today()
        end = today + pd.Timedelta(days=120)
        req = CorporateActionsRequest(
            symbols=[ticker],
            types=["earnings"],
            start=today,
            end=end.to_pydatetime().date() if hasattr(end, "to_pydatetime") else today,
        )
        resp = client.get_corporate_actions(req)
        rows = getattr(resp, "data", None) or {}
        items = rows.get("earnings", []) if isinstance(rows, dict) else []
        dates = []
        for item in items:
            d = getattr(item, "ex_date", None) or getattr(item, "date", None)
            if isinstance(d, datetime):
                d = d.date()
            if isinstance(d, date) and d >= today:
                dates.append(d)
        return min(dates) if dates else None
    except Exception as exc:
        logger.debug(f"Alpaca earnings lookup failed for {ticker}: {exc}")
        return None


def _yfinance_next_earnings(ticker: str) -> date | None:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    try:
        cal = yf.Ticker(ticker).get_calendar()
    except Exception as exc:
        logger.debug(f"yfinance earnings lookup failed for {ticker}: {exc}")
        return None
    if cal is None:
        return None

    today = date.today()
    candidates: list[date] = []
    if isinstance(cal, dict):
        ed = cal.get("Earnings Date")
        if isinstance(ed, list):
            for v in ed:
                if isinstance(v, datetime):
                    candidates.append(v.date())
                elif isinstance(v, date):
                    candidates.append(v)
    elif hasattr(cal, "loc"):
        try:
            ed = cal.loc["Earnings Date"]
            if hasattr(ed, "values"):
                for v in ed.values:
                    if isinstance(v, datetime):
                        candidates.append(v.date())
        except Exception:
            pass

    upcoming = [d for d in candidates if d >= today]
    return min(upcoming) if upcoming else None
