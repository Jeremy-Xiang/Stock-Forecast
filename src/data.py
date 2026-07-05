"""
data.py — Load OHLC price history for a ticker.

Primary path: pull live data via yfinance.
Fallback path: generate a reproducible synthetic OHLC series so the rest
of the pipeline (models, backtest, plots) can be developed and tested
without network access. Synthetic data is clearly labeled as such and
should never be used to draw real conclusions about a stock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlc(ticker: str = "AAPL", period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """
    Return a DataFrame indexed by date with columns: Open, High, Low, Close, Volume.

    Tries yfinance first. If that fails (no network, rate limited, etc.),
    falls back to a synthetic series so the code still runs end-to-end.
    """
    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError("yfinance returned no data")

        # yfinance sometimes returns a MultiIndex on columns (ticker, field)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"
        return df

    except Exception as exc:  # noqa: BLE001 - we want a clean fallback for any failure
        print(f"[data.py] Live fetch failed ({exc}). Falling back to synthetic data for testing.")
        return _synthetic_ohlc(n_days=2500, seed=_stable_seed(ticker))


def _stable_seed(ticker: str) -> int:
    """
    Deterministic seed from a ticker string. sum(ord(c) for c in ticker)
    looks deterministic too, but collides on any anagram (e.g. 'GS' and
    'KO' both sum to 154), which would silently give two different tickers
    identical synthetic price paths. crc32 over the actual byte sequence
    avoids that.
    """
    import zlib

    return zlib.crc32(ticker.encode()) % (2**32)


def _synthetic_ohlc(n_days: int = 2500, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    """
    Generate a plausible OHLC series via geometric Brownian motion plus
    a small intraday range, purely for offline development/testing.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    daily_returns = rng.normal(loc=0.0004, scale=0.018, size=n_days)
    close = start_price * np.exp(np.cumsum(daily_returns))

    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.003, size=n_days - 1))

    intraday_range = np.abs(rng.normal(0.006, 0.004, size=n_days)) * close
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range
    volume = rng.integers(2_000_000, 90_000_000, size=n_days)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df
