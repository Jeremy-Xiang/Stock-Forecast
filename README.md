# stock-forecast-bench

A walk-forward comparison of four one-step-ahead OHLC forecasting approaches,
runnable on a single ticker or across a whole basket. Built around a
transition-matrix idea from an earlier exploratory notebook, cleaned up and
put on a level playing field against three standard baselines, then scaled
from "one stock" to "every stock you actually care about."

This is an educational/portfolio project, **not** a trading signal. Predicting
next-day closing prices from price history alone is a famously hard problem;
the point here is the methodology — clean walk-forward evaluation, transparent
math, an honest comparison, and a basket mode that tells you whether a result
generalizes or was just luck on one ticker.

## The four models

| Model | Idea |
|---|---|
| **Naive (persistence)** | Tomorrow's OHLC = today's OHLC. The standard random-walk baseline every other model has to beat. |
| **Transition matrix** | A linear dynamical system fit to the whole training window (see math below). |
| **Linear Regression** | Multi-output linear regression on a small lag-feature set. |
| **Random Forest** | Multi-output random forest on the same lag-feature set. |

### The transition matrix model

Represent each day as a state vector of its four prices:

```
x_t = [Open_t, High_t, Low_t, Close_t]   ∈ R^4
```

The model assumes a *fixed linear map* `T` (a 4×4 matrix) that approximately
advances the state by one day:

```
x_(t+1) ≈ x_t · T
```

Given a training window with `n` days, stack the states into two matrices —
`X0` (days `0..n-2`) and `X1` (days `1..n-1`, i.e. `X0` shifted forward by one
day). The best-fit `T` minimizes:

```
‖X1 − X0·T‖²
```

which is solved directly with ordinary least squares (`numpy.linalg.lstsq`).
This is the cleaned-up version of the original notebook's approach, which
estimated many small local 4×4 transition matrices (one per 4-day window) by
explicitly inverting each `X0` block and averaging the results — numerically
fragile, since some of those small windows are close to singular. Here we
solve for the single best `T` across the *entire* training set in one stable
least-squares fit. Same underlying idea (a Markov-style linear transition
between consecutive states), estimated more robustly.

Once fit, a forecast is `x_t · T`. For a `k`-day-ahead forecast, apply the
matrix power: `x_t · T^k` — the part of the original notebook that was
already a genuine Markov-chain technique (repeated application of a fixed
transition operator) carries over unchanged
(`TransitionMatrixModel.predict_k_steps`).

### Linear Regression & Random Forest

Both use the same hand-built feature set: the flattened OHLC values from the
last 5 trading days, plus the 5-day rolling mean and standard deviation of
the close. The target is next-day OHLC.

## Evaluation methodology

All four models are fit **once**, on the first 80% of the price history
(`train_frac=0.8`, configurable). They are then evaluated **one step ahead**
on the remaining 20%: every test-day forecast uses only true price history
from strictly before that day. No model is refit during the test window.

Metrics reported:
- **MAE / RMSE** on closing price (in dollars)
- **Directional accuracy** — did the model get the up/down move right,
  relative to the previous day's *actual* close?

> **Note on directional accuracy:** naive persistence always predicts "no
> change," so its predicted direction is flat (0), which can never match a
> real, non-zero daily move. It will show ~0% directional accuracy *by
> construction*, not because it's an unusually bad model. MAE/RMSE are the
> fair comparison for that baseline.

## Two real bugs found while building this

Worth documenting since they're the kind of thing that's easy to get subtly
wrong in any walk-forward backtest, not just this one:

1. **O(n²) backtest loop.** The first version called `predict_next()` once
   per test day, and for the regression-based models that meant rebuilding
   the entire lag-feature matrix from scratch on every call. Fine for one
   ticker, painfully slow across a basket (~3 minutes for 4 tickers). Fixed
   by recognizing that none of the four models ever feed a prior prediction
   back in as the next day's input — they're all pure functions of *true*
   history — so the whole test window can be forecast in a single vectorized
   batch call (`predict_series()`) instead of a day-by-day loop.
2. **Off-by-one feature alignment.** While restructuring for the fix above, a
   second, independent bug turned up: `predict_next()`'s feature-building
   step required a known "next day" target to exist in the data it was
   given, which it doesn't for an actual live forecast — so it was silently
   producing a prediction *for the last day already in history*, using lag
   data from one day further back than intended. It happened to validate
   internally (it ran without error, returned a plausible-looking price) — it
   just wasn't predicting the day it claimed to. This only affected Linear
   Regression and Random Forest (Naive and the transition matrix model index
   directly into the last row and never had this problem). Fixed with a
   dedicated `build_latest_feature_vector()` that doesn't require a target
   row. Verified by checking `predict_next()` and `predict_series()` agree
   exactly on overlapping dates (they now match to floating-point precision).

The lesson generalizes: a model that runs cleanly and returns a
reasonable-looking number is not the same as a model that's predicting the
day you think it's predicting. Worth an explicit equality check between any
"live" and "batch" prediction path before trusting either.

## Running it

```bash
pip install -r requirements.txt

# Single ticker — detailed per-model prediction plots
python run_comparison.py --tickers AAPL

# Basket mode — cross-ticker ranking (2+ tickers triggers this automatically)
python run_comparison.py --tickers AAPL,MSFT,GOOGL,AMZN,JPM,XOM,JNJ,TSLA

# Built-in sector-diverse default basket (13 tickers)
python run_comparison.py --basket

# Your own list from a file, one ticker per line — e.g. your full
# 53-ticker THESIS universe
python run_comparison.py --tickers-file my_tickers.txt
```

This pulls live daily OHLC data via `yfinance`. If it can't reach the
network (e.g. a sandboxed CI environment), `src/data.py` falls back to a
reproducible synthetic OHLC series so the pipeline still runs end to end —
useful for testing, never for real conclusions about a stock.

**Runtime:** Random Forest's `fit()` is the dominant cost (~5-6s per
ticker on ~2,000 training rows with 300 trees). Budget roughly that per
ticker — a 53-ticker basket takes on the order of 5 minutes. Reduce
`n_estimators` in `src/models.py` if you want faster iteration at some cost
to RF accuracy.

## Single-ticker mode output

```
Model                                    MAE ($)  RMSE ($)   Dir. Acc.
----------------------------------------------------------------------
Naive (persistence)                        1.222     1.538        0.0%
Transition matrix (linear dynamical system) 1.223     1.540       48.6%
Linear Regression (lag features)           1.229     1.541       49.2%
Random Forest (lag features)               1.850     2.618       48.4%
```

Saves `plots/model_comparison.png` (actual vs. predicted close, per model)
and `plots/metric_comparison.png` (bar charts across all four).

## Basket mode output

```
Model                                    Avg MAE  Avg RMSE  Avg Dir.Acc  Avg Rank  Win Rate
-------------------------------------------------------------------------------------------
Naive (persistence)                        4.363     5.527         0.0%      1.54     46.2%
Transition matrix (linear dynamical system) 4.363     5.532        49.7%      1.62     46.2%
Linear Regression (lag features)           4.400     5.570        50.1%      2.85      7.7%
Random Forest (lag features)              24.442    30.764        49.2%      4.00      0.0%
```

Saves `plots/basket_results.csv` (every ticker × model result, for your own
slicing), `plots/basket_mae_boxplot.png` (MAE spread per model across the
whole basket), and `plots/basket_win_rate.png` (how often each model had the
lowest MAE for a given ticker).

**Takeaways that hold up across runs, single-ticker or basket:**
- Naive persistence and the transition matrix track each other almost
  exactly on MAE/RMSE — a single global linear map over OHLC space ends up
  close to "today predicts tomorrow" for a near-random-walk series like
  daily stock prices. The transition matrix's value-add is directional edge
  and clean multi-step (`T^k`) forecasting, not point-forecast accuracy.
- Random Forest's MAE/RMSE are consistently the worst, and its *spread*
  across tickers is wide — a handful of tickers with strong trends blow up
  its error (see `basket_mae_boxplot.png`). This is a structural property:
  tree-based models can't extrapolate beyond the price range they were
  trained on. When a stock moves past its training-period high or low, RF's
  prediction flattens out near that boundary instead of following the price
  (visible directly in `plots/model_comparison.png`'s bottom-right panel).
  Worth knowing before reaching for RF on any trending series.
- More parameters didn't buy more accuracy here. That's the textbook outcome
  for next-day stock price forecasting from price history alone — a
  genuinely useful negative result, not a bug.

## Project structure

```
stock-forecast-bench/
├── run_comparison.py      # entry point — single-ticker or basket mode
├── src/
│   ├── data.py            # yfinance loader (+ offline synthetic fallback)
│   ├── models.py          # the four models
│   ├── backtest.py        # walk-forward evaluation + metrics
│   └── aggregate.py       # cross-ticker basket runner + ranking
├── plots/                 # generated comparison plots + basket_results.csv
└── requirements.txt
```

## Serving this from a dashboard (cache-first API)

`app.py` wraps the backtest in a FastAPI service designed for exactly the
constraint a free-tier host like Render puts on you: Random Forest's
`fit()` takes ~5-6s per ticker, which is too slow to redo on every page
load, especially after a cold start.

The pattern:
- An APScheduler cron job (`precompute_tickers()`) runs once a day,
  computing all four models for every ticker in your universe and writing
  the result to a small JSON cache (`src/cache.py`).
- Every `GET /forecast/{ticker}` request reads from that cache — no model
  fitting on the request path.
- First request for a ticker that isn't cached yet (or whose cache entry
  is stale) computes it once, synchronously, caches it, and every
  subsequent request for that ticker is a cache hit until the next
  scheduled refresh. Slow once, fast after — the standard cache warm-up
  tradeoff, not a special case to handle.

```bash
uvicorn app:app --reload --port 8002
curl http://localhost:8002/health
curl http://localhost:8002/forecast/AAPL
curl -X POST http://localhost:8002/admin/refresh -H "Content-Type: application/json" -d '["AAPL","MSFT"]'
```

`/forecast/{ticker}` returns JSON (dates + actual/predicted close + metrics
per model) — built for a `ComposedChart` in Recharts, not for re-parsing a
matplotlib PNG.

To mount this under THESIS's existing FastAPI app instead of running a
second service: copy `src/cache.py`, `src/precompute.py`, and the route
handlers in `app.py` in as a router (`app.include_router(...)`), point
`DEFAULT_TICKERS` at your real 53-ticker universe via the
`FORECAST_TICKERS` env var, and reuse THESIS's existing scheduler if it
already runs one (one cron scheduler per process is plenty — no need for
this module's `BackgroundScheduler` *and* THESIS's APScheduler instance
both running).

## Running the tests

```bash
pytest tests/ -v
```

The suite pins the behaviors that actually caught bugs during development
(see the sections above), not ceremony coverage — every test encodes a
check where the wrong answer was at some point the actual behavior.

## Possible next steps

- Wire this into a proper backtest module inside the THESIS dashboard
  (Sharpe ratio, max drawdown, per-theme breakdown) rather than a standalone
  script.
- Add a Markov *regime* classifier (e.g. via clustering rolling volatility
  and trend features into discrete states) and fit a separate transition
  matrix per regime, instead of one matrix for the whole series.
- Extend the lag-feature set with the sentiment scores already computed in
  THESIS, to see whether they actually move the needle on the regression
  models' error.
