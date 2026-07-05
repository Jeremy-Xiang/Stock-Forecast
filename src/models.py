"""
models.py — Four comparable one-step-ahead OHLC forecasting models.

State vector convention: x_t = [Open_t, High_t, Low_t, Close_t] (in that order).

1. NaivePersistenceModel   — predicts tomorrow == today (random-walk baseline)
2. TransitionMatrixModel   — linear dynamical system x_{t+1} ≈ x_t @ T, fit by
                              least squares. This is the cleaned-up version of
                              the "transition matrix" approach from the
                              original exploratory notebook.
3. LinearRegressionModel   — multi-output linear regression on lag features
4. RandomForestModel       — multi-output random forest on lag features

All four models only ever use TRUE historical prices as input — none of them
feed a prior prediction back in as the next day's input. That means a whole
backtest window can be forecast in a single vectorized call instead of a
day-by-day loop, which matters a lot once you're running this across dozens
of tickers rather than one.

Each model implements:
    fit(df) -> None
    predict_next(history) -> np.ndarray, shape (4,)      # one day, for live use
    predict_series(df) -> (preds, dates)                  # vectorized, for backtesting
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

OHLC = ["Open", "High", "Low", "Close"]
N_LAGS = 5  # how many prior days of OHLC feed the regression-based models


def build_lag_features(df: pd.DataFrame, n_lags: int = N_LAGS):
    """
    Build a supervised-learning table from a price history:
      X[i] = flattened OHLC for the n_lags days ending at day i,
             plus 5-day rolling mean/std of Close as of day i
      y[i] = OHLC on day i+1 (the target we're forecasting)

    Returns (X, y, index) where index aligns to the *target* day (i+1),
    i.e. dates[k] is the day X[k]/y[k] is forecasting.
    """
    ohlc = df[OHLC].to_numpy()
    n = len(df)

    roll_mean = df["Close"].rolling(5).mean().to_numpy()
    roll_std = df["Close"].rolling(5).std().to_numpy()

    rows = []
    targets = []
    idx = []
    for i in range(n_lags, n - 1):
        if np.isnan(roll_mean[i]) or np.isnan(roll_std[i]):
            continue
        lag_block = ohlc[i - n_lags + 1 : i + 1].flatten()  # n_lags * 4 values
        extra = [roll_mean[i], roll_std[i]]
        rows.append(np.concatenate([lag_block, extra]))
        targets.append(ohlc[i + 1])
        idx.append(df.index[i + 1])

    X = np.array(rows)
    y = np.array(targets)
    return X, y, pd.DatetimeIndex(idx)


def build_latest_feature_vector(df: pd.DataFrame, n_lags: int = N_LAGS) -> np.ndarray:
    """
    Build the single feature row for forecasting the day AFTER df's last row.

    This is intentionally separate from build_lag_features(), which requires
    a known "next day" target to exist in the data and therefore can never
    construct a feature row for a day that hasn't happened yet. Using
    build_lag_features(history)[-1] to approximate "predict tomorrow" is an
    easy off-by-one mistake: its last constructible row targets history's
    OWN last day (using lag data from one day further back), not the day
    after history ends.
    """
    if len(df) < n_lags + 5:
        raise ValueError(f"Need at least {n_lags + 5} rows of history, got {len(df)}.")

    ohlc = df[OHLC].to_numpy()
    lag_block = ohlc[-n_lags:].flatten()
    roll_mean = df["Close"].iloc[-5:].mean()
    roll_std = df["Close"].iloc[-5:].std()
    return np.concatenate([lag_block, [roll_mean, roll_std]]).reshape(1, -1)


class NaivePersistenceModel:
    """Baseline: tomorrow's OHLC = today's OHLC. No fitting required."""

    name = "Naive (persistence)"

    def fit(self, df: pd.DataFrame) -> None:
        return  # nothing to learn

    def predict_next(self, history: pd.DataFrame) -> np.ndarray:
        return history[OHLC].iloc[-1].to_numpy()

    def predict_series(self, df: pd.DataFrame):
        X = df[OHLC].to_numpy()
        preds = X[:-1]  # prediction FOR day i+1 is day i's actual OHLC
        dates = df.index[1:]
        return preds, dates


class TransitionMatrixModel:
    """
    Linear dynamical system: x_{t+1} ≈ x_t @ T, where x_t = [O, H, L, C].

    T is fit once over the whole training window via ordinary least squares
    (np.linalg.lstsq), which is the numerically stable equivalent of the
    "average local 4x4 transition matrix" idea from the original notebook —
    instead of inverting many small, sometimes near-singular 4x4 blocks and
    averaging the results, we solve the single best-fit T directly.
    """

    name = "Transition matrix (linear dynamical system)"

    def __init__(self):
        self.T: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> None:
        X = df[OHLC].to_numpy()
        X0 = X[:-1]  # state at t
        X1 = X[1:]  # state at t+1
        # Solve X1 ≈ X0 @ T  =>  T = lstsq(X0, X1)
        T, *_ = np.linalg.lstsq(X0, X1, rcond=None)
        self.T = T

    def predict_next(self, history: pd.DataFrame) -> np.ndarray:
        x_t = history[OHLC].iloc[-1].to_numpy()
        return x_t @ self.T

    def predict_k_steps(self, history: pd.DataFrame, k: int) -> np.ndarray:
        """Multi-day-ahead forecast via T^k, matching the notebook's original use case."""
        x_t = history[OHLC].iloc[-1].to_numpy()
        T_k = np.linalg.matrix_power(self.T, k)
        return x_t @ T_k

    def predict_series(self, df: pd.DataFrame):
        X = df[OHLC].to_numpy()
        preds = X[:-1] @ self.T
        dates = df.index[1:]
        return preds, dates


class LinearRegressionModel:
    name = "Linear Regression (lag features)"

    def __init__(self, n_lags: int = N_LAGS):
        self.n_lags = n_lags
        self.model = LinearRegression()

    def fit(self, df: pd.DataFrame) -> None:
        X, y, _ = build_lag_features(df, self.n_lags)
        self.model.fit(X, y)

    def predict_next(self, history: pd.DataFrame) -> np.ndarray:
        tail = history.tail(self.n_lags + 5)
        X = build_latest_feature_vector(tail, self.n_lags)
        return self.model.predict(X)[0]

    def predict_series(self, df: pd.DataFrame):
        X, _, dates = build_lag_features(df, self.n_lags)
        preds = self.model.predict(X)
        return preds, dates


class RandomForestModel:
    name = "Random Forest (lag features)"

    def __init__(self, n_lags: int = N_LAGS, n_estimators: int = 300, random_state: int = 42):
        self.n_lags = n_lags
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    def fit(self, df: pd.DataFrame) -> None:
        X, y, _ = build_lag_features(df, self.n_lags)
        self.model.fit(X, y)
        # n_jobs=-1 is worth it for the one-time fit (parallel tree training).
        # predict_series below makes one batched predict() call, so the
        # per-call joblib dispatch overhead that hurt the old day-by-day
        # loop doesn't apply here regardless of this setting.
        self.model.n_jobs = 1

    def predict_next(self, history: pd.DataFrame) -> np.ndarray:
        tail = history.tail(self.n_lags + 5)
        X = build_latest_feature_vector(tail, self.n_lags)
        return self.model.predict(X)[0]

    def predict_series(self, df: pd.DataFrame):
        X, _, dates = build_lag_features(df, self.n_lags)
        preds = self.model.predict(X)  # one batched call instead of hundreds of single-row ones
        return preds, dates


def get_all_models():
    return [
        NaivePersistenceModel(),
        TransitionMatrixModel(),
        LinearRegressionModel(),
        RandomForestModel(),
    ]
