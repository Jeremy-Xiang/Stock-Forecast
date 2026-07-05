# stock-forecast-bench

Compares four next-day OHLC forecasting models side by side under strict walk-forward validation — single ticker or a whole basket. The baseline is naive persistence (tomorrow = today), and every other model has to beat it. Two of them don't.

## The four models

Four models, one interface (`fit`, `predict_next`, `predict_series`):

- **Naive persistence** — tomorrow's OHLC = today's. Every other model is measured against this.
- **Transition matrix** — a 4×4 linear map fit by least squares, treating [Open, High, Low, Close] as a state vector. `x_(t+1) ≈ x_t · T`, where `T` minimizes `‖X1 − X0·T‖²`. The original notebook computed many small local matrices and averaged them — numerically fragile because some 4-day windows are near-singular. Here we solve for one `T` across the full training window. Multi-day forecasts via `T^k`.
- **Linear regression** — multi-output OLS on 5 days of lagged OHLC plus a 5-day rolling mean/std of close.
- **Random forest** — same feature set. 300 trees, `max_depth=7`, time-series cross-validation.

## How evaluation works

All four models train once on the first 80% of price history. The remaining 20% is evaluated one step ahead: each forecast uses only the true price history before that day, never anything from the test window. No refitting during evaluation.

Metrics: MAE and RMSE on closing price (dollars), and directional accuracy (did the model call the daily move direction correctly). Naive persistence always predicts zero change, so its directional accuracy is 0% by construction — not because it's bad, just because "no change" can't match a real non-zero move. Use MAE/RMSE to compare it.

## Results

```
Model                                    MAE ($)  RMSE ($)   Dir. Acc.
----------------------------------------------------------------------
Naive (persistence)                        1.222     1.538        0.0%
Transition matrix (linear dynamical system) 1.223     1.540       48.6%
Linear Regression (lag features)           1.229     1.541       49.2%
Random Forest (lag features)               1.850     2.618       48.4%
```

Basket mode adds cross-ticker ranking columns (average MAE rank, win rate per model). Saves `basket_results.csv` for your own slicing.

The result that holds across every run: Random Forest is consistently the worst. Tree-based models can't extrapolate past their training range — when a stock trends outside the prices it was trained on, RF predictions flatline at the boundary. This shows up as the wide error bars in `basket_mae_boxplot.png` and is visible directly in `model_comparison.png`'s bottom-right panel. It's a structural property of the model class, not a tuning issue.

## Two bugs found during development

**O(n²) backtest loop.** The first version called `predict_next()` once per test day. For the regression models, each call rebuilt the entire lag-feature matrix from scratch, turning a single-ticker backtest into an O(n²) operation. Three minutes for four tickers. Fixed by recognizing that none of the four models feed prior predictions back as input — they're all pure functions of true history — so the test window can be forecast in one vectorized batch call (`predict_series()`) instead of a loop.

**Off-by-one feature alignment.** While restructuring for the fix above, a second bug surfaced: `predict_next()`'s feature builder required a known "next day" target row to exist in the data it was given. For an actual live forecast, there is no next day yet — so it was silently forecasting the *last day already in history*, using lag data from a day further back than intended. It ran without errors and returned plausible-looking prices. That's the failure mode: not a crash, just the wrong day. Fixed with `build_latest_feature_vector()`, which constructs a feature row from the raw tail of history without needing a target. Verified by checking `predict_next()` and `predict_series()` agree to floating-point precision on overlapping dates.

A model that runs cleanly and returns a plausible number is not the same as a model predicting the day you think it is.

## Running it

```bash
pip install -r requirements.txt

python run_comparison.py --tickers AAPL               # single ticker, per-model plots
python run_comparison.py --tickers AAPL,MSFT,XOM,JNJ  # basket mode, cross-ticker ranking
python run_comparison.py --basket                      # built-in 13-ticker universe
python run_comparison.py --tickers-file my_tickers.txt # your own list, one per line
```

yfinance pulls live OHLCV data. If the network's unavailable (sandboxed CI, etc.), `src/data.py` falls back to a deterministic synthetic series labeled as such.

Random Forest's `fit()` dominates runtime — about 5-6s per ticker on 2,000 training rows. A 53-ticker basket takes roughly 5 minutes. Drop `n_estimators` in `src/models.py` if you need faster iteration.

## Cache-first API

`app.py` is a FastAPI wrapper for use cases where the results need to be served over HTTP (e.g. a React dashboard). Random Forest is too slow to refit per request, so an APScheduler cron job precomputes every ticker nightly and writes results to a JSON cache. `GET /forecast/{ticker}` reads from cache; the first request for an uncached ticker computes synchronously, then caches for all subsequent requests.

```bash
uvicorn app:app --port 8002
curl http://localhost:8002/forecast/AAPL
curl -X POST http://localhost:8002/admin/refresh -d '["AAPL","MSFT"]'
```

Response shape is `{dates, actual_close, predicted_close, mae, rmse}` per model — ready for a Recharts `ComposedChart`.

## Running the tests

```bash
pytest tests/ -v
```

Six tests. Every one encodes a check where the wrong answer was, at some point, the actual behavior.

## Structure

```
stock-forecast-bench/
├── run_comparison.py   # CLI: single-ticker or basket mode
├── app.py              # FastAPI service with caching layer
├── src/
│   ├── models.py       # the four models
│   ├── backtest.py     # walk-forward evaluation
│   ├── aggregate.py    # cross-ticker basket runner + ranking
│   ├── cache.py        # JSON cache for the API
│   ├── precompute.py   # nightly precompute job
│   └── data.py         # yfinance loader + synthetic fallback
└── tests/test_core.py
```
