"""Telegram push notifications for trade events."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
from loguru import logger

_ET = ZoneInfo("America/New_York")


def _journal_path() -> Path:
    return Path(os.getenv("TRADE_JOURNAL_FILE", "logs/trade_journal.jsonl"))


def journal_event(entry: dict) -> None:
    """Append a trade/rebalance/CB event to the shared JSONL journal.

    Both bot services write here; the daily report reads it back to answer
    "what did you trade today and why".
    """
    try:
        path = _journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(tz=_ET).isoformat(timespec="seconds"), **entry}
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning(f"Trade journal write failed: {exc}")


def read_journal(date_iso: str) -> list[dict]:
    """Return journal entries whose ET timestamp falls on the given ISO date."""
    path = _journal_path()
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(entry.get("ts", "")).startswith(date_iso):
                entries.append(entry)
    return entries


def _format_event(entry: dict) -> str:
    event = entry.get("event")
    bot = entry.get("bot", "?")
    reason = entry.get("reason", "")
    if event == "trade":
        emoji = "🟢" if entry.get("action") == "BUY" else "🔴"
        line = f"{emoji} [{bot}] {entry.get('action')} {entry.get('symbol')} ${entry.get('value', 0):,.0f}"
    elif event == "rebalance":
        line = f"📊 [{bot}] Rebalance → {entry.get('target')}"
    elif event == "circuit_breaker":
        line = f"🚨 [{bot}] Circuit breaker at {entry.get('drawdown', 0):.1%}"
    else:
        line = f"[{bot}] {event}"
    if reason:
        line += f" — {reason}"
    return line


class TelegramNotifier:
    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.token and self.chat_id)
        if not self._enabled:
            logger.warning(
                "Telegram notifications DISABLED: TELEGRAM_BOT_TOKEN and/or "
                "TELEGRAM_CHAT_ID missing from environment"
            )

    def send(self, text: str) -> bool:
        """Send a message. Returns True only on confirmed delivery to Telegram."""
        if not self._enabled:
            logger.warning("Telegram message NOT sent (notifier disabled)")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = httpx.post(
                url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}, timeout=10
            )
            body = resp.json()
            if not body.get("ok"):
                logger.error(f"Telegram API rejected message: {body}")
                return False
            return True
        except Exception as exc:
            logger.warning(f"Telegram notify failed: {exc}")
            return False

    def trade(self, bot: str, action: str, symbol: str, value: float, reason: str = "") -> None:
        journal_event({"event": "trade", "bot": bot, "action": action, "symbol": symbol,
                       "value": value, "reason": reason})
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        msg = f"{emoji} <b>[{bot}] {action} {symbol}</b>\nValue: ${value:,.0f}"
        if reason:
            msg += f"\nReason: {reason}"
        self.send(msg)

    def rebalance(self, bot: str, target: str, regime: str, portfolio_value: float, detail: str = "") -> None:
        journal_event({"event": "rebalance", "bot": bot, "target": target, "regime": regime,
                       "value": portfolio_value, "reason": detail})
        msg = f"📊 <b>[{bot}] Rebalance → {target}</b>\nRegime: {regime}\nPortfolio: ${portfolio_value:,.0f}"
        if detail:
            msg += f"\n{detail}"
        self.send(msg)

    def circuit_breaker(self, bot: str, drawdown: float, portfolio_value: float) -> None:
        journal_event({"event": "circuit_breaker", "bot": bot, "drawdown": drawdown,
                       "value": portfolio_value, "reason": "drawdown limit hit, moved to cash/stable"})
        msg = (
            f"🚨 <b>[{bot}] CIRCUIT BREAKER TRIGGERED</b>\n"
            f"Drawdown: {drawdown:.1%}\nMoved to cash/stable\nValue: ${portfolio_value:,.0f}"
        )
        self.send(msg)

    def proposal(
        self,
        proposal_id: str,
        strategy_id: str,
        params: dict,
        hypothesis: str,
        expected: dict,
        shadow_summary: str,
    ) -> None:
        """Strategy-promotion proposal from the evolution agent. Approval is a
        human CLI action on the droplet; this message carries the exact command."""
        journal_event({"event": "evolve_proposal", "bot": "Evolve",
                       "proposal_id": proposal_id, "reason": hypothesis})
        param_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(params.items()))
        msg = (
            f"🧬 <b>[Evolve] Promotion proposal {proposal_id}</b>\n"
            f"Strategy: {strategy_id}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Params:\n{param_lines}\n"
            f"Backtest expectation: CAGR {expected.get('cagr', 0):+.1%}, "
            f"Sharpe {expected.get('sharpe', 0):.2f}, MaxDD {expected.get('max_dd', 0):+.1%}\n"
            f"Shadow: {shadow_summary}\n"
            f"Note: shadow validates plumbing and the drawdown envelope, not alpha — "
            f"statistical confidence comes from the backtest + holdout.\n\n"
            f"Approve:  python main.py evolve approve {proposal_id}\n"
            f"Reject:   python main.py evolve reject {proposal_id}"
        )
        self.send(msg)

    def startup(self, bot: str) -> None:
        self.send(f"✅ <b>[{bot}]</b> Bot started on DigitalOcean")

    def health(self, bot: str, portfolio_value: float, holding: Optional[str], detail: str = "") -> None:
        msg = f"💓 <b>[{bot}] Daily health</b>\nHolding: {holding or 'CASH'}\nValue: ${portfolio_value:,.0f}"
        if detail:
            msg += f"\n{detail}"
        self.send(msg)

    def daily_report(
        self,
        date_iso: str,
        equity: float,
        last_equity: float,
        start_capital: float,
        cash: float,
        positions: list[tuple[str, float, float]],
        bot_lines: list[str],
        events: list[dict],
    ) -> str:
        """Post-market daily report: account value, holdings, and today's trades with reasons.

        positions: (symbol, market_value, unrealized_pnl) tuples.
        Returns the message text (also sent), so callers/tests can inspect it.
        """
        day_pnl = equity - last_equity
        day_pct = day_pnl / last_equity if last_equity > 0 else 0.0
        total_pnl = equity - start_capital
        total_pct = total_pnl / start_capital if start_capital > 0 else 0.0
        lines = [
            f"📋 <b>Daily report — {date_iso}</b>",
            f"Account: ${equity:,.0f} ({day_pnl:+,.0f} / {day_pct:+.2%} today)",
            f"Since start: {total_pnl:+,.0f} ({total_pct:+.2%} on ${start_capital:,.0f})",
            "",
            "<b>Holdings</b>",
        ]
        for symbol, market_value, unrealized_pnl in positions:
            lines.append(f"• {symbol}: ${market_value:,.0f} (P&L {unrealized_pnl:+,.0f})")
        lines.append(f"• Cash: ${cash:,.0f}")
        if bot_lines:
            lines.append("")
            lines.append("<b>Bots</b>")
            lines.extend(f"• {b}" for b in bot_lines)
        lines.append("")
        lines.append("<b>Today's activity</b>")
        if events:
            lines.extend(f"• {_format_event(e)}" for e in events)
        else:
            lines.append("• No trades today")
        msg = "\n".join(lines)
        self.send(msg)
        return msg
