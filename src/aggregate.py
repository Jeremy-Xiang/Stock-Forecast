"""
aggregate.py — Run the four-model walk-forward backtest across a basket of
tickers and summarize which model wins where.

A single ticker tells you almost nothing about whether a model is actually
good — it might just be lucky on that one stock's particular path. Running
the same harness across a basket (ideally a few dozen tickers spanning
different sectors/volatility regimes) is what turns this into a real
comparison instead of an anecdote.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import run_backtest
from .data import load_ohlc
from .models import get_all_models

# A small, sector-diverse default basket. Swap in your own list (e.g. your
# 53 THESIS tickers) via --tickers or --tickers-file on the CLI.
DEFAULT_BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # tech / mega-cap growth
    "JPM", "BAC",                              # financials
    "XOM", "CVX",                              # energy
    "JNJ", "PG", "KO",                         # defensive / consumer staples
    "TSLA",                                    # high-volatility growth
]


@dataclass
class TickerLoadFailure:
    ticker: str
    reason: str


def run_basket_backtest(tickers: list[str], period: str = "10y", train_frac: float = 0.8):
    """
    Returns (results_df, failures) where results_df has one row per
    (ticker, model) with columns: ticker, model, mae, rmse, directional_accuracy.
    """
    rows = []
    failures: list[TickerLoadFailure] = []

    for ticker in tickers:
        try:
            df = load_ohlc(ticker, period=period)
            if len(df) < 50:
                failures.append(TickerLoadFailure(ticker, "not enough history"))
                continue
        except Exception as exc:  # noqa: BLE001
            failures.append(TickerLoadFailure(ticker, str(exc)))
            continue

        for model in get_all_models():
            try:
                result = run_backtest(model, df, train_frac=train_frac)
            except Exception as exc:  # noqa: BLE001
                # one bad model/ticker combo shouldn't kill the whole basket run
                print(f"  [skip] {ticker} / {model.name}: {exc}")
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "model": result.model_name,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "directional_accuracy": result.directional_accuracy,
                }
            )

    return pd.DataFrame(rows), failures


def rank_models(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, rank the models by MAE (1 = best/lowest error on that
    ticker). Returns a per-model summary: average rank, win rate (fraction
    of tickers where that model had the lowest MAE), and average metrics.
    """
    ranked = results_df.copy()
    ranked["mae_rank"] = ranked.groupby("ticker")["mae"].rank(method="min")

    summary = (
        ranked.groupby("model")
        .agg(
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_directional_accuracy=("directional_accuracy", "mean"),
            avg_mae_rank=("mae_rank", "mean"),
            win_count=("mae_rank", lambda s: int((s == 1).sum())),
            n_tickers=("ticker", "nunique"),
        )
        .sort_values("avg_mae_rank")
    )
    return summary


def plot_basket_summary(results_df: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> list[str]:
    saved = []

    # Box plot: spread of MAE per model across all tickers
    fig, ax = plt.subplots(figsize=(9, 5))
    models = summary.index.tolist()
    data = [results_df.loc[results_df["model"] == m, "mae"].to_numpy() for m in models]
    ax.boxplot(data, tick_labels=models, showmeans=True)
    ax.set_ylabel("MAE ($) across tickers")
    ax.set_title("MAE distribution by model, across the ticker basket")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path1 = f"{out_dir}/basket_mae_boxplot.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    saved.append(path1)

    # Bar chart: win rate per model (fraction of tickers where it had the lowest MAE)
    fig, ax = plt.subplots(figsize=(9, 5))
    win_rate = summary["win_count"] / summary["n_tickers"] * 100
    ax.bar(win_rate.index, win_rate.to_numpy(), color="tab:purple")
    ax.set_ylabel("Win rate (%) — lowest MAE on that ticker")
    ax.set_title("How often each model is the best one for a given stock")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path2 = f"{out_dir}/basket_win_rate.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    saved.append(path2)

    return saved
