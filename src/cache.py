"""
cache.py — Turn BacktestResult objects into JSON, and cache them to disk
with a timestamp so the API can serve cheap reads instead of recomputing
Random Forest fits on every request.

File-based cache rather than a database because: (a) this needs to run on
Render's free tier without an extra paid add-on, and (b) the cached payload
is small (a few KB per ticker) and fully rebuilt on each scheduled run, so
there's no migration/schema concern — just read it, and if it's missing or
stale, rebuild it.

Caveat: Render's free tier filesystem is ephemeral across deploys (not
across requests/restarts within the same deploy). If you're on a plan
where the disk doesn't persist at all, point CACHE_PATH at a small SQLite
file on a Render persistent disk, or just accept that the first request
after each deploy is a cold-compute and gets cached for everything after.
Either way this module doesn't need to change — only where CACHE_PATH points.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .backtest import BacktestResult

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "forecast_cache.json")


def result_to_dict(result: BacktestResult) -> dict:
    return {
        "model_name": result.model_name,
        "dates": [d.strftime("%Y-%m-%d") for d in result.dates],
        "actual_close": [float(x) for x in result.actual_close],
        "predicted_close": [float(x) for x in result.predicted_close],
        "mae": result.mae,
        "rmse": result.rmse,
        "directional_accuracy": result.directional_accuracy,
    }


def load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {"computed_at": None, "tickers": {}}
    with open(CACHE_PATH) as f:
        return json.load(f)


def save_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)


def get_cached_ticker(ticker: str, max_age_hours: float = 24.0) -> Optional[dict]:
    """Return the cached entry for a ticker if present and not stale, else None."""
    cache = load_cache()
    entry = cache.get("tickers", {}).get(ticker)
    if entry is None:
        return None

    computed_at = cache.get("computed_at")
    if computed_at is None:
        return None

    age_hours = (time.time() - computed_at) / 3600
    if age_hours > max_age_hours:
        return None

    return entry


def update_cache(ticker_results: dict[str, list[dict]]) -> None:
    """
    ticker_results: {ticker: [result_dict, result_dict, ...]} — one dict per
    model, already run through result_to_dict().
    """
    cache = load_cache()
    cache["computed_at"] = time.time()
    cache.setdefault("tickers", {})
    cache["tickers"].update(ticker_results)
    save_cache(cache)
