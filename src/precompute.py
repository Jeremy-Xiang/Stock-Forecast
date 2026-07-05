"""
precompute.py — The actual "expensive work," run on a schedule rather than
per-request. Computes all four models for every ticker in the configured
universe and writes the result to the cache.

This is the function APScheduler calls on a cron trigger. It's also exposed
as a plain function (not tied to FastAPI or the scheduler) so it can be
tested or run manually from a script without spinning up a server.
"""

from __future__ import annotations

from .backtest import run_backtest
from .cache import result_to_dict, update_cache
from .data import load_ohlc
from .models import get_all_models


def precompute_tickers(tickers: list[str], period: str = "10y", train_frac: float = 0.8) -> dict:
    """Run the full model comparison for each ticker and write to cache. Returns what was written."""
    ticker_results = {}

    for ticker in tickers:
        try:
            df = load_ohlc(ticker, period=period)
            if len(df) < 50:
                print(f"[precompute] Skipping {ticker}: not enough history ({len(df)} rows)")
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"[precompute] Skipping {ticker}: {exc}")
            continue

        model_results = []
        for model in get_all_models():
            try:
                result = run_backtest(model, df, train_frac=train_frac)
                model_results.append(result_to_dict(result))
            except Exception as exc:  # noqa: BLE001
                print(f"[precompute] {ticker} / {model.name} failed: {exc}")

        if model_results:
            ticker_results[ticker] = model_results
            print(f"[precompute] {ticker}: cached {len(model_results)} model results")

    if ticker_results:
        update_cache(ticker_results)

    return ticker_results
