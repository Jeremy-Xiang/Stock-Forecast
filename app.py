"""
app.py — FastAPI service for THESIS forecast integration.

The pattern: an APScheduler cron job runs precompute_tickers() once a day
(configurable), writing results to the JSON cache. Every API request reads
from that cache — no model fitting happens on the request path, so response
times don't depend on Random Forest's ~5-6s-per-ticker fit cost or on
Render's free-tier CPU limits.

First request after a fresh deploy (empty cache) triggers a synchronous
compute as a fallback so the endpoint doesn't just 404 — slow once, then
fast for everyone after, same tradeoff as any cache warm-up.

Run standalone for local development:
    uvicorn app:app --reload --port 8002

Mount under THESIS's existing FastAPI app for production (see README).
"""

from __future__ import annotations

import os
import secrets

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.cache import get_cached_ticker, load_cache
from src.precompute import precompute_tickers

# Same default basket as run_comparison.py's --basket flag. In production,
# swap this for your real 53-ticker THESIS universe (env var, config file,
# or a call into THESIS's existing ticker list rather than a constant here).
DEFAULT_TICKERS = os.environ.get("FORECAST_TICKERS", "AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,XOM,JNJ,PG,TSLA").split(",")
PRECOMPUTE_HOUR = int(os.environ.get("FORECAST_PRECOMPUTE_HOUR", "2"))  # 2am server time, low-traffic

# Comma-separated allowlist of browser origins, e.g. "https://thesis.jeremyxiang.com".
# Falls back to "*" for local dev; set it in production to lock the API to your frontend.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# Shared secret guarding /admin/*. Unset => admin routes are disabled (fail closed),
# so an anonymous caller can never kick off an expensive recompute.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")


def _require_admin(token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin endpoints disabled: set ADMIN_TOKEN to enable them.")
    if not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token.")

scheduler = BackgroundScheduler()

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Startup — same lifespan pattern THESIS's own main.py uses
    # (FastAPI has deprecated @app.on_event in favor of this).
    scheduler.add_job(
        precompute_tickers,
        CronTrigger(hour=PRECOMPUTE_HOUR),
        args=[DEFAULT_TICKERS],
        id="nightly_forecast_precompute",
        replace_existing=True,
    )
    scheduler.start()

    # Warm the cache immediately on startup if it's empty, so the very
    # first deploy doesn't serve 404s until 2am.
    cache = load_cache()
    if not cache.get("tickers"):
        print("[app] Cache empty on startup — running an initial precompute now.")
        precompute_tickers(DEFAULT_TICKERS)

    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="THESIS Forecast API", version="1.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    cache = load_cache()
    return {
        "status": "ok",
        "cached_tickers": list(cache.get("tickers", {}).keys()),
        "computed_at": cache.get("computed_at"),
    }


@app.get("/forecast/{ticker}")
def get_forecast(ticker: str, max_age_hours: float = 24.0):
    ticker = ticker.upper()
    cached = get_cached_ticker(ticker, max_age_hours=max_age_hours)

    if cached is None:
        # Cold path: nothing cached (or it's stale) for this ticker. Compute
        # it once, synchronously, cache it, and return — every subsequent
        # request for this ticker is then a cache hit until the next
        # scheduled refresh.
        print(f"[app] Cache miss for {ticker} — computing synchronously.")
        results = precompute_tickers([ticker])
        cached = results.get(ticker)
        if cached is None:
            raise HTTPException(status_code=404, detail=f"Could not compute a forecast for '{ticker}'.")

    return {"ticker": ticker, "models": cached}


@app.post("/admin/refresh")
def refresh(tickers: list[str] | None = None, x_admin_token: str | None = Header(default=None)):
    """Manually trigger a recompute — useful for a 'refresh data' button in the UI."""
    _require_admin(x_admin_token)
    target = [t.upper() for t in tickers] if tickers else DEFAULT_TICKERS
    results = precompute_tickers(target)
    return {"refreshed": list(results.keys())}
