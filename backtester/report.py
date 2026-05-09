from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from tabulate import tabulate

from backtester.metrics import compare_to_benchmark


class BacktestReport:
    def __init__(self, portfolio_values: pd.Series, benchmark_values: pd.Series):
        self.portfolio = portfolio_values
        self.benchmark = benchmark_values
        self.metrics = compare_to_benchmark(portfolio_values, benchmark_values)

    def print_summary(self) -> None:
        m = self.metrics
        if not m:
            print("Insufficient data for metrics.")
            return

        rows = [
            ["Metric", "Strategy", "SPY Benchmark"],
            ["CAGR", f"{m['cagr']:.1%}", f"{m['benchmark_cagr']:.1%}"],
            ["Total Return", f"{m['total_return']:.1%}", "—"],
            ["Ann. Volatility", f"{m['ann_vol']:.1%}", "—"],
            ["Sharpe Ratio", f"{m['sharpe']:.2f}", f"{m['benchmark_sharpe']:.2f}"],
            ["Sortino Ratio", f"{m['sortino']:.2f}", "—"],
            ["Max Drawdown", f"{m['max_drawdown']:.1%}", f"{m['benchmark_max_drawdown']:.1%}"],
            ["Calmar Ratio", f"{m['calmar']:.2f}", "—"],
            ["Alpha (ann.)", f"{m['alpha']:.1%}", "—"],
            ["Beta", f"{m['beta']:.2f}", "1.00"],
            ["Info. Ratio", f"{m['information_ratio']:.2f}", "—"],
            ["Monthly Win Rate vs SPY", f"{m['win_rate_vs_benchmark']:.1%}", "—"],
            ["Years", f"{m['n_years']:.1f}", "—"],
        ]

        header = rows.pop(0)
        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
        print()

        # Flag if strategy doesn't beat benchmark
        if m["cagr"] <= m["benchmark_cagr"]:
            print("⚠  Strategy DOES NOT beat SPY in this backtest period.")
            print("   Review signal weights and consider a different strategy before proceeding.")
        else:
            excess = m["cagr"] - m["benchmark_cagr"]
            print(f"✓  Strategy beats SPY by {excess:.1%} annualized (CAGR).")

    def plot(self, output_path: str = "backtest_results.png") -> None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle("Backtest Results: Strategy vs SPY", fontsize=14, fontweight="bold")

        # Normalize to 100
        port_norm = self.portfolio / self.portfolio.iloc[0] * 100
        bench_norm = self.benchmark.reindex(self.portfolio.index).ffill()
        bench_norm = bench_norm / bench_norm.iloc[0] * 100

        # Panel 1: Equity curves
        ax = axes[0]
        ax.plot(port_norm.index, port_norm.values, label="Strategy", color="#2196F3", linewidth=1.5)
        ax.plot(bench_norm.index, bench_norm.values, label="SPY", color="#FF9800", linewidth=1.5, alpha=0.8)
        ax.set_ylabel("Portfolio Value (base=100)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Equity Curve")

        # Panel 2: Drawdown
        ax = axes[1]
        rolling_max = self.portfolio.cummax()
        drawdown = (self.portfolio - rolling_max) / rolling_max * 100
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.6)
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Strategy Drawdown")
        ax.grid(True, alpha=0.3)

        # Panel 3: Rolling 12-month returns vs benchmark
        ax = axes[2]
        port_ret = self.portfolio.pct_change()
        bench_ret = self.benchmark.reindex(self.portfolio.index).ffill().pct_change()
        rolling_alpha = (port_ret - bench_ret).rolling(252).sum() * 100
        ax.bar(rolling_alpha.index, rolling_alpha.values,
               color=["#4CAF50" if v >= 0 else "#F44336" for v in rolling_alpha.values],
               width=1, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Rolling 12M Alpha (%)")
        ax.set_title("Rolling Alpha vs SPY")
        ax.grid(True, alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
        plt.close()


def plot_feature_importance(
    importances: "pd.Series",
    output_path: str = "feature_importance.png",
) -> None:
    """Horizontal bar chart of XGBoost feature importances (gain)."""
    import pandas as pd

    fig, ax = plt.subplots(figsize=(8, 5))
    imp = importances.sort_values(ascending=True)

    colors = ["#2196F3" if v >= imp.median() else "#90CAF9" for v in imp.values]
    bars = ax.barh(imp.index, imp.values, color=colors, edgecolor="white", height=0.6)

    # Label each bar
    for bar, val in zip(bars, imp.values):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Feature Importance (gain)", fontsize=11)
    ax.set_title("XGBoost Feature Importances\n(final trained model)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, imp.max() * 1.20)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Feature importance chart saved to {output_path}")
    plt.close()
