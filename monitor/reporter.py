from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from backtester.metrics import compute_metrics

if TYPE_CHECKING:
    from execution.orders import Order


class DailyReporter:
    """Generates daily performance summaries to console and a dated log file."""

    def __init__(self, initial_capital: float = 5000.0, log_dir: str = "logs"):
        self.initial_capital = initial_capital
        self._log_dir = Path(log_dir) / "daily"
        self._history: list[dict] = []

    def record(self, portfolio_value: float, positions: pd.DataFrame, date: datetime) -> None:
        self._history.append(
            {"date": date, "value": portfolio_value, "n_positions": len(positions)}
        )

    def daily_summary(
        self,
        portfolio_value: float,
        positions: pd.DataFrame,
        benchmark_value: float | None = None,
        scores: pd.Series | None = None,
        orders: list[Order] | None = None,
        risk_events: list[str] | None = None,
        drawdown: float = 0.0,
        circuit_breaker: bool = False,
    ) -> None:
        today = datetime.now()
        pnl = portfolio_value - self.initial_capital
        pnl_pct = pnl / self.initial_capital

        lines = [
            "",
            "=" * 54,
            f"  DAILY SUMMARY  {today.strftime('%Y-%m-%d %H:%M')}",
            "=" * 54,
            f"  Portfolio Value : ${portfolio_value:>10,.2f}",
            f"  Total P&L       : ${pnl:>+10,.2f}  ({pnl_pct:>+.1%})",
        ]

        if benchmark_value is not None:
            bench_ret = benchmark_value / self.initial_capital - 1
            alpha = pnl_pct - bench_ret
            lines.append(f"  vs SPY          : {bench_ret:>+.1%}  (alpha {alpha:>+.1%})")

        lines.append(f"  Current DD      : {drawdown:.1%}")
        lines.append(f"  Positions       : {len(positions)}")

        # Risk events
        if risk_events:
            lines.append("")
            lines.append("  Risk Events:")
            for event in risk_events:
                lines.append(f"    ! {event}")
        elif circuit_breaker:
            lines.append("    ! Circuit breaker ACTIVE — rebalancing suspended")

        # Top signal scores
        if scores is not None and not scores.empty:
            lines.append("")
            lines.append("  Top Signals:")
            for ticker, score in scores.nlargest(10).items():
                lines.append(f"    {ticker:<6}  score={score:.4f}")

        # Orders submitted
        if orders:
            lines.append("")
            lines.append(f"  Orders ({len(orders)} MOC):")
            for o in orders:
                lines.append(f"    {o.side.value.upper():<4}  {o.ticker:<6}  ${o.notional:>9,.0f}")

        # Current holdings
        if not positions.empty and "market_value" in positions.columns:
            lines.append("")
            lines.append("  Top Holdings:")
            top = positions.nlargest(5, "market_value")
            for ticker, row in top.iterrows():
                pnl_str = f"{row.get('unrealized_pnl', 0):>+.0f}"
                lines.append(f"    {ticker:<6}  ${row['market_value']:>8,.0f}  PnL: ${pnl_str}")

        lines.append("=" * 54)

        output = "\n".join(lines)
        print(output)
        logger.info(f"Daily summary: value=${portfolio_value:,.2f}, positions={len(positions)}")

        self.record(portfolio_value, positions, today)
        self._write_to_file(output, today)

    def _write_to_file(self, content: str, date: datetime) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"{date.strftime('%Y-%m-%d')}.log"
        with open(path, "a") as f:
            f.write(content + "\n")
        logger.debug(f"Daily summary written to {path}")

    def intraday_summary(
        self,
        trades: list[dict],
        portfolio_value: float,
        benchmark_value: float | None = None,
        daily_start_value: float | None = None,
        candidates: list | None = None,
    ) -> None:
        today = datetime.now()
        start = daily_start_value or self.initial_capital
        daily_pnl = portfolio_value - start
        daily_pnl_pct = daily_pnl / start if start > 0 else 0.0

        wins  = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        win_rate = len(wins) / len(trades) if trades else 0.0

        lines = [
            "",
            "=" * 54,
            f"  INTRADAY SUMMARY  {today.strftime('%Y-%m-%d')}",
            "=" * 54,
            f"  Portfolio Value : ${portfolio_value:>10,.2f}",
            f"  Daily P&L       : ${daily_pnl:>+10,.2f}  ({daily_pnl_pct:>+.1%})",
        ]

        if benchmark_value is not None:
            bench_ret = benchmark_value / self.initial_capital - 1
            alpha = daily_pnl_pct - bench_ret
            lines.append(f"  vs SPY          : {bench_ret:>+.1%}  (alpha {alpha:>+.1%})")

        lines += [
            f"  Trades Today    : {len(trades)}  ({len(wins)}W / {len(losses)}L)  "
            f"win rate {win_rate:.0%}",
            f"  Gross P&L       : ${total_pnl:>+,.2f}",
        ]

        if wins:
            best = max(wins, key=lambda t: t.get("pnl", 0))
            lines.append(f"  Best Trade      : {best['ticker']} +${best['pnl']:,.2f}")
        if losses:
            worst = min(losses, key=lambda t: t.get("pnl", 0))
            lines.append(f"  Worst Trade     : {worst['ticker']} ${worst['pnl']:,.2f}")

        if trades:
            lines.append("")
            lines.append("  Trades:")
            for t in trades:
                pnl = t.get("pnl", 0)
                ctype = t.get("close_type", "")
                lines.append(
                    f"    {t['ticker']:<6}  {t.get('side','').upper():<5}  "
                    f"entry={t.get('entry', 0):.2f}  "
                    f"exit={t.get('exit', 0):.2f}  "
                    f"P&L=${pnl:>+.2f}  [{ctype}]"
                )

        if candidates:
            lines.append("")
            lines.append("  Candidates scanned:")
            for c in candidates[:8]:
                lines.append(
                    f"    {c.ticker:<6}  {c.direction:<5}  "
                    f"gap={c.gap_pct:>+.1%}  score={c.gap_score:.3f}"
                )

        lines.append("=" * 54)
        output = "\n".join(lines)
        print(output)
        logger.info(
            f"Intraday summary: {len(trades)} trades, daily P&L={daily_pnl_pct:+.1%}"
        )
        self._write_to_file(output, today)
