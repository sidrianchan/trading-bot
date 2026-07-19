"""Tests for the trade journal and post-market daily Telegram report."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from monitor.notify import TelegramNotifier, journal_event, read_journal


@pytest.fixture
def journal_file(tmp_path, monkeypatch):
    path = tmp_path / "trade_journal.jsonl"
    monkeypatch.setenv("TRADE_JOURNAL_FILE", str(path))
    return path


def _today_et() -> str:
    return datetime.now(tz=ZoneInfo("America/New_York")).date().isoformat()


class TestJournal:
    def test_write_and_read_today(self, journal_file):
        journal_event({"event": "trade", "bot": "ETF", "action": "BUY",
                       "symbol": "SOXL", "value": 70_000.0, "reason": "risk_on, top momentum"})
        entries = read_journal(_today_et())
        assert len(entries) == 1
        assert entries[0]["symbol"] == "SOXL"
        assert entries[0]["ts"].startswith(_today_et())

    def test_read_filters_other_dates(self, journal_file):
        journal_file.write_text(
            '{"ts": "2020-01-01T10:00:00-05:00", "event": "trade", "bot": "ETF"}\n'
        )
        assert read_journal(_today_et()) == []
        assert len(read_journal("2020-01-01")) == 1

    def test_read_missing_file(self, journal_file):
        assert read_journal(_today_et()) == []

    def test_read_skips_corrupt_lines(self, journal_file):
        journal_file.write_text("not json\n")
        journal_event({"event": "trade", "bot": "Crypto", "action": "SELL",
                       "symbol": "ETH/USD", "value": 100.0})
        assert len(read_journal(_today_et())) == 1

    def test_notifier_methods_journal_without_telegram_creds(self, journal_file, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        notify = TelegramNotifier()
        notify.trade("ETF", "BUY", "SOXL", 70_000.0, "risk_on")
        notify.rebalance("Crypto", "ETH/USD", "risk_on", 30_000.0, "ETH best 7d momentum")
        notify.circuit_breaker("ETF", -0.344, 57_211.0)
        events = read_journal(_today_et())
        assert [e["event"] for e in events] == ["trade", "rebalance", "circuit_breaker"]


class TestDailyReport:
    def _notifier(self, monkeypatch) -> TelegramNotifier:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        return TelegramNotifier()

    def test_report_with_trades(self, monkeypatch):
        notify = self._notifier(monkeypatch)
        msg = notify.daily_report(
            "2026-07-20",
            equity=93_192.0,
            last_equity=93_604.0,
            start_capital=100_000.0,
            cash=63_342.0,
            positions=[("USDC", 29_850.0, -12.0)],
            bot_lines=["ETF: CASH — circuit breaker cooldown (1/3 risk-on months)", "Crypto: USDC"],
            events=[
                {"event": "trade", "bot": "Crypto", "action": "BUY", "symbol": "ETH/USD",
                 "value": 29_900.0, "reason": "fill@1761.20"},
                {"event": "rebalance", "bot": "Crypto", "target": "ETH/USD",
                 "reason": "BTC risk-on; ETH best 7d relative momentum"},
            ],
        )
        assert "Daily report — 2026-07-20" in msg
        assert "$93,192" in msg
        assert "-412" in msg          # day P&L
        assert "-6,808" in msg        # since-start P&L
        assert "USDC: $29,850" in msg
        assert "circuit breaker cooldown" in msg
        assert "BUY ETH/USD $29,900 — fill@1761.20" in msg
        assert "Rebalance → ETH/USD — BTC risk-on; ETH best 7d relative momentum" in msg

    def test_report_no_trades(self, monkeypatch):
        notify = self._notifier(monkeypatch)
        msg = notify.daily_report(
            "2026-07-20", 100_000.0, 100_000.0, 100_000.0, 100_000.0,
            positions=[], bot_lines=[], events=[],
        )
        assert "No trades today" in msg
        assert "+0.00%" in msg
