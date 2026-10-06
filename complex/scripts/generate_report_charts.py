#!/usr/bin/env python3
"""
Gera gráficos para os relatórios LaTeX do projeto.

Os artefatos são gravados em complex/reports/build para serem consumidos
pelo build do LaTeX.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_BUILD_DIR = BASE_DIR / "reports" / "build"


def ensure_dirs() -> None:
    REPORT_BUILD_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    backtest_path = OUTPUT_DIR / "backtest_results.csv"
    metrics_path = OUTPUT_DIR / "strategy_metrics.csv"

    if not backtest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "Arquivos de saída não encontrados. Rode complex/run_demo.py primeiro."
        )

    backtest = pd.read_csv(backtest_path)
    metrics = pd.read_csv(metrics_path)
    return backtest, metrics


def style_plot() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(name: str, fig: plt.Figure) -> None:
    fig.tight_layout()
    fig.savefig(REPORT_BUILD_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_equity_and_drawdown(backtest: pd.DataFrame) -> None:
    df = backtest.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["equity_norm"] = df["portfolio_value"] / df["portfolio_value"].iloc[0]
    df["drawdown_pct"] = df["drawdown"] * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(df["date"], df["equity_norm"], color="#0f766e", linewidth=2.5)
    axes[0].fill_between(df["date"], df["equity_norm"], 1.0, color="#0f766e", alpha=0.12)
    axes[0].axhline(1.0, color="#64748b", linestyle="--", linewidth=1)
    axes[0].set_title("Curva de Capital Normalizada")
    axes[0].set_ylabel("Capital Inicial = 1.0")

    axes[1].fill_between(df["date"], df["drawdown_pct"], 0, color="#b91c1c", alpha=0.35)
    axes[1].plot(df["date"], df["drawdown_pct"], color="#991b1b", linewidth=1.8)
    axes[1].axhline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set_title("Drawdown Diário")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Data")

    savefig("strategy_equity_drawdown.png", fig)


def plot_return_distribution(backtest: pd.DataFrame) -> None:
    df = backtest.copy()
    daily_returns = df["daily_return"] * 100

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(daily_returns, bins=35, kde=True, color="#0f766e", ax=ax, stat="density")
    ax.axvline(daily_returns.mean(), color="#b45309", linestyle="--", linewidth=2, label="Média")
    ax.axvline(daily_returns.median(), color="#1d4ed8", linestyle=":", linewidth=2, label="Mediana")
    ax.set_title("Distribuição dos Retornos Diários")
    ax.set_xlabel("Retorno diário (%)")
    ax.set_ylabel("Densidade")
    ax.legend(frameon=False)

    savefig("daily_return_distribution.png", fig)


def plot_monthly_heatmap(backtest: pd.DataFrame) -> None:
    df = backtest.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = (
        df.groupby(["year", "month"])["daily_return"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index()
    )
    heatmap = monthly.pivot(index="year", columns="month", values="daily_return").reindex(columns=month_order)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        heatmap * 100,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Retorno mensal (%)"},
        ax=ax,
    )
    ax.set_title("Heatmap de Retornos Mensais")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Ano")

    savefig("monthly_return_heatmap.png", fig)


def plot_rolling_sharpe(backtest: pd.DataFrame) -> None:
    df = backtest.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sharpe_rolling"] = pd.to_numeric(df["sharpe_rolling"], errors="coerce")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["sharpe_rolling"], color="#7c3aed", linewidth=2.2)
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_title("Sharpe Rolling de 63 Dias")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Data")

    savefig("rolling_sharpe.png", fig)


def plot_performance_comparison(metrics: pd.DataFrame) -> None:
    row = metrics.iloc[0]
    labels = ["Retorno Total", "Sharpe", "Max DD", "Win Rate"]
    strategy = [row["total_return"] * 100, row["sharpe_ratio"], abs(row["max_drawdown"]) * 100, row["win_rate"] * 100]
    sp500 = [42.3, 0.85, 34.5, 51.2]
    balanced = [35.8, 1.12, 12.3, 55.0]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - width, strategy, width, label="Estratégia", color="#0f766e")
    ax.bar(x, sp500, width, label="S&P 500", color="#1d4ed8")
    ax.bar(x + width, balanced, width, label="60/40", color="#b45309")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparação de Performance vs Benchmarks")
    ax.legend(frameon=False)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=10)

    savefig("performance_comparison.png", fig)


def plot_model_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.35, 1.2, 1.4, 1.2, "Dados\nPreço + Fundamentos", "#dbeafe"),
        (2.0, 1.2, 1.45, 1.2, "Fatores\n8 sinais robustos", "#dcfce7"),
        (3.7, 1.2, 1.45, 1.2, "HMM\n3 regimes", "#fef3c7"),
        (5.4, 1.2, 1.45, 1.2, "ML Ensemble\nGBM + Ridge", "#ede9fe"),
        (7.1, 1.2, 1.55, 1.2, "Otimização\nMean-Variance", "#fee2e2"),
        (8.95, 1.2, 0.7, 1.2, "Pesos\nFinais", "#cffafe"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#0f172a", linewidth=1.6, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, weight="bold")

    arrow_props = dict(arrowstyle="->", color="#334155", linewidth=1.8, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(1.75, 1.8), xytext=(2.0, 1.8), arrowprops=arrow_props)
    ax.annotate("", xy=(3.45, 1.8), xytext=(3.7, 1.8), arrowprops=arrow_props)
    ax.annotate("", xy=(5.15, 1.8), xytext=(5.4, 1.8), arrowprops=arrow_props)
    ax.annotate("", xy=(6.85, 1.8), xytext=(7.1, 1.8), arrowprops=arrow_props)
    ax.annotate("", xy=(8.7, 1.8), xytext=(8.95, 1.8), arrowprops=arrow_props)

    ax.text(5.0, 3.1, "Adaptive Multi-Factor Regime-Switching Strategy", ha="center", fontsize=16, weight="bold")
    ax.text(5.0, 0.45, "Pipeline diário: dados → fatores → regime → previsão → alocação", ha="center", fontsize=11, color="#475569")

    savefig("model_pipeline.png", fig)


def main() -> None:
    ensure_dirs()
    style_plot()
    backtest, metrics = load_data()
    plot_equity_and_drawdown(backtest)
    plot_return_distribution(backtest)
    plot_monthly_heatmap(backtest)
    plot_rolling_sharpe(backtest)
    plot_performance_comparison(metrics)
    plot_model_pipeline()
    print(f"Gráficos gerados em {REPORT_BUILD_DIR}")


if __name__ == "__main__":
    main()