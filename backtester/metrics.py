from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(portfolio_values: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Compute standard performance metrics from a daily portfolio value series."""
    returns = portfolio_values.pct_change().dropna()

    if returns.empty or len(returns) < 2:
        return {}

    n_years = len(returns) / 252
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate) / downside if downside > 0 else 0.0

    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "cagr": cagr,
        "total_return": total_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "n_years": n_years,
    }


def compare_to_benchmark(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series,
    risk_free_rate: float = 0.05,
) -> dict:
    """Compute alpha, beta, and information ratio vs a benchmark."""
    port_ret = portfolio_values.pct_change().dropna()
    bench_ret = benchmark_values.pct_change().dropna()

    common = port_ret.index.intersection(bench_ret.index)
    if len(common) < 30:
        return {}

    p = port_ret.loc[common]
    b = bench_ret.loc[common]

    cov_matrix = np.cov(p, b)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.0

    port_metrics = compute_metrics(portfolio_values.loc[common], risk_free_rate)
    bench_metrics = compute_metrics(benchmark_values.loc[common], risk_free_rate)

    # Jensen's alpha (annualized)
    bench_cagr = bench_metrics.get("cagr", 0.0)
    port_cagr = port_metrics.get("cagr", 0.0)
    alpha = port_cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

    active_returns = p - b
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0.0

    # Monthly win rate vs benchmark
    port_monthly = (1 + p).resample("ME").prod() - 1
    bench_monthly = (1 + b).resample("ME").prod() - 1
    common_months = port_monthly.index.intersection(bench_monthly.index)
    win_rate = (port_monthly.loc[common_months] > bench_monthly.loc[common_months]).mean()

    return {
        **port_metrics,
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "win_rate_vs_benchmark": win_rate,
        "benchmark_cagr": bench_cagr,
        "benchmark_sharpe": bench_metrics.get("sharpe", 0.0),
        "benchmark_max_drawdown": bench_metrics.get("max_drawdown", 0.0),
    }
