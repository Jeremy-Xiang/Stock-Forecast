"""
Tests for stock-forecast-bench. Run: pytest tests/ -v

These encode the checks that actually caught bugs during development —
not ceremony tests. The off-by-one feature-alignment bug (predict_next
silently forecasting the wrong day) only surfaced because live and batch
prediction paths were compared for exact equality; that check is now
permanent here.
"""

import numpy as np
import pytest

from src.backtest import run_backtest, train_test_split_by_date
from src.data import _stable_seed, _synthetic_ohlc
from src.models import get_all_models, OHLC


@pytest.fixture(scope="module")
def price_df():
    return _synthetic_ohlc(n_days=800, seed=42)


def test_stable_seed_no_anagram_collisions():
    # sum(ord(c)) collides on anagrams (GS/KO both -> 154); crc32 must not.
    assert _stable_seed("GS") != _stable_seed("KO")
    tickers = ["AAPL", "MSFT", "GOOGL", "GS", "KO", "PG", "XOM", "JNJ"]
    seeds = [_stable_seed(t) for t in tickers]
    assert len(set(seeds)) == len(tickers)


def test_synthetic_data_is_deterministic():
    a = _synthetic_ohlc(n_days=100, seed=7)
    b = _synthetic_ohlc(n_days=100, seed=7)
    assert (a["Close"] == b["Close"]).all()


def test_predict_next_matches_predict_series(price_df):
    """The off-by-one bug detector: live single-day prediction must equal
    the batch path's prediction FOR THE SAME DATE, exactly."""
    train, test = train_test_split_by_date(price_df, 0.8)
    cutoff = test.index[10]
    history = price_df.loc[price_df.index < cutoff]

    for model in get_all_models():
        model.fit(train)
        next_pred = model.predict_next(history)
        preds, dates = model.predict_series(price_df)
        idx = list(dates).index(cutoff)
        assert np.max(np.abs(next_pred - preds[idx])) < 1e-9, model.name


def test_backtest_no_lookahead(price_df):
    """Every predicted date must be inside the test window — a prediction
    dated inside the training window means leakage."""
    train, test = train_test_split_by_date(price_df, 0.8)
    for model in get_all_models():
        result = run_backtest(model, price_df, train_frac=0.8)
        assert result.dates.min() >= test.index[0], model.name
        assert len(result.actual_close) == len(result.predicted_close)


def test_backtest_metrics_sane(price_df):
    for model in get_all_models():
        r = run_backtest(model, price_df, train_frac=0.8)
        assert r.mae > 0
        assert r.rmse >= r.mae  # RMSE >= MAE always
        assert 0.0 <= r.directional_accuracy <= 1.0


def test_cache_roundtrip(tmp_path, monkeypatch):
    from src import cache

    monkeypatch.setattr(cache, "CACHE_PATH", str(tmp_path / "c.json"))
    assert cache.get_cached_ticker("AAPL") is None
    cache.update_cache({"AAPL": [{"model_name": "x", "mae": 1.0}]})
    hit = cache.get_cached_ticker("AAPL")
    assert hit is not None and hit[0]["mae"] == 1.0
    assert cache.get_cached_ticker("AAPL", max_age_hours=0) is None  # stale
