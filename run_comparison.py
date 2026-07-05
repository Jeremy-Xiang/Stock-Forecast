"""
run_comparison.py — Entry point. Loads AAPL data, runs all four models
through the walk-forward backtest, prints a metrics table, and saves two
comparison plots to ./plots/.

Usage:
    python run_comparison.py [--ticker AAPL] [--period 10y] [--train-frac 0.8]
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.aggregate import DEFAULT_BASKET, plot_basket_summary, rank_models, run_basket_backtest
from src.backtest import run_backtest
from src.data import load_ohlc
from src.models import get_all_models

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def print_metrics_table(results) -> None:
    header = f"{'Model':<38}{'MAE ($)':>10}{'RMSE ($)':>10}{'Dir. Acc.':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.model_name:<38}{r.mae:>10.3f}{r.rmse:>10.3f}{r.directional_accuracy * 100:>11.1f}%")


def plot_predictions(results, ticker: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for ax, r in zip(axes, results):
        ax.plot(r.dates, r.actual_close, label="Actual", color="black", linewidth=1.2)
        ax.plot(r.dates, r.predicted_close, label="Predicted", color="tab:red", linewidth=1.0, alpha=0.8)
        ax.set_title(
            f"{r.model_name}\n"
            f"MAE=\\${r.mae:.2f}   RMSE=\\${r.rmse:.2f}   Dir.Acc={r.directional_accuracy*100:.1f}%"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(axis="x", labelrotation=30)

    fig.suptitle(f"{ticker}: Actual vs Predicted Close — Walk-Forward Test Window", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_bars(results, ticker: str) -> str:
    names = [r.model_name for r in results]
    maes = [r.mae for r in results]
    rmses = [r.rmse for r in results]
    dir_acc = [r.directional_accuracy * 100 for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(names, maes, color="tab:blue")
    axes[0].set_title("MAE ($, lower is better)")
    axes[0].tick_params(axis="x", labelrotation=25)

    axes[1].bar(names, rmses, color="tab:orange")
    axes[1].set_title("RMSE ($, lower is better)")
    axes[1].tick_params(axis="x", labelrotation=25)

    axes[2].bar(names, dir_acc, color="tab:green")
    axes[2].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[2].set_title("Directional Accuracy (%, higher is better)")
    axes[2].tick_params(axis="x", labelrotation=25)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_fontsize(8)

    fig.suptitle(f"{ticker}: Model Comparison Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(PLOTS_DIR, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_single_ticker(ticker: str, period: str, train_frac: float) -> None:
    df = load_ohlc(ticker, period=period)
    print(f"Loaded {len(df)} rows for {ticker}: {df.index.min().date()} to {df.index.max().date()}\n")

    results = []
    for model in get_all_models():
        result = run_backtest(model, df, train_frac=train_frac)
        results.append(result)

    print_metrics_table(results)

    p1 = plot_predictions(results, ticker)
    p2 = plot_metric_bars(results, ticker)
    print(f"\nSaved: {p1}")
    print(f"Saved: {p2}")


def run_basket(tickers: list[str], period: str, train_frac: float) -> None:
    print(f"Running {len(get_all_models())} models across {len(tickers)} tickers...")
    print(f"Tickers: {', '.join(tickers)}\n")

    results_df, failures = run_basket_backtest(tickers, period=period, train_frac=train_frac)

    if failures:
        print("Skipped (could not load or not enough history):")
        for f in failures:
            print(f"  - {f.ticker}: {f.reason}")
        print()

    if results_df.empty:
        print("No results — every ticker failed to load. Check tickers and network access.")
        return

    summary = rank_models(results_df)

    print(f"{'Model':<38}{'Avg MAE':>10}{'Avg RMSE':>10}{'Avg Dir.Acc':>13}{'Avg Rank':>10}{'Win Rate':>10}")
    print("-" * 91)
    for model_name, row in summary.iterrows():
        win_rate = row["win_count"] / row["n_tickers"] * 100
        print(
            f"{model_name:<38}{row['avg_mae']:>10.3f}{row['avg_rmse']:>10.3f}"
            f"{row['avg_directional_accuracy'] * 100:>12.1f}%{row['avg_mae_rank']:>10.2f}"
            f"{win_rate:>9.1f}%"
        )

    csv_path = os.path.join(PLOTS_DIR, "basket_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-ticker results: {csv_path}")

    saved_plots = plot_basket_summary(results_df, summary, PLOTS_DIR)
    for p in saved_plots:
        print(f"Saved: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="AAPL",
        help="Comma-separated list, e.g. 'AAPL,MSFT,GOOGL'. Single ticker runs in detailed mode "
        "(per-model prediction plots); 2+ tickers run in basket mode (cross-ticker ranking).",
    )
    parser.add_argument(
        "--tickers-file",
        default=None,
        help="Optional path to a text file with one ticker per line. Use this for large baskets "
        "(e.g. your full THESIS 53-ticker list) instead of a long --tickers string.",
    )
    parser.add_argument("--basket", action="store_true", help="Force basket mode using the built-in default basket.")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    elif args.basket:
        tickers = DEFAULT_BASKET
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if len(tickers) == 1:
        run_single_ticker(tickers[0], args.period, args.train_frac)
    else:
        run_basket(tickers, args.period, args.train_frac)


if __name__ == "__main__":
    main()
