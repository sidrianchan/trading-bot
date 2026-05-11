"""Telegram push notifications for trade events."""
from __future__ import annotations

import os
from typing import Optional

import httpx
from loguru import logger


class TelegramNotifier:
    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.token and self.chat_id)

    def send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            httpx.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        except Exception as exc:
            logger.warning(f"Telegram notify failed: {exc}")

    def trade(self, bot: str, action: str, symbol: str, value: float, reason: str = "") -> None:
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        msg = f"{emoji} <b>[{bot}] {action} {symbol}</b>\nValue: ${value:,.0f}"
        if reason:
            msg += f"\nReason: {reason}"
        self.send(msg)

    def rebalance(self, bot: str, target: str, regime: str, portfolio_value: float, detail: str = "") -> None:
        msg = f"📊 <b>[{bot}] Rebalance → {target}</b>\nRegime: {regime}\nPortfolio: ${portfolio_value:,.0f}"
        if detail:
            msg += f"\n{detail}"
        self.send(msg)

    def circuit_breaker(self, bot: str, drawdown: float, portfolio_value: float) -> None:
        msg = (
            f"🚨 <b>[{bot}] CIRCUIT BREAKER TRIGGERED</b>\n"
            f"Drawdown: {drawdown:.1%}\nMoved to cash/stable\nValue: ${portfolio_value:,.0f}"
        )
        self.send(msg)

    def startup(self, bot: str) -> None:
        self.send(f"✅ <b>[{bot}]</b> Bot started on DigitalOcean")

    def health(self, bot: str, portfolio_value: float, holding: Optional[str], detail: str = "") -> None:
        msg = f"💓 <b>[{bot}] Daily health</b>\nHolding: {holding or 'CASH'}\nValue: ${portfolio_value:,.0f}"
        if detail:
            msg += f"\n{detail}"
        self.send(msg)
