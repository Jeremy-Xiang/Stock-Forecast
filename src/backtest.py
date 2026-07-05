"""
backtest.py — Walk-forward, one-step-ahead evaluation for the four models.

Each model is fit ONCE on the training window only (no peeking at test data
during fitting). Forecasts for the test window are then computed in a single
vectorized batch via `model.predict_series(df)`, which is safe here because
none of the four models ever feed a prior prediction back in as input — they
all forecast day t+1 purely from the TRUE price history up through day t.
That means there's no autoregressive feedback loop to simulate day-by-day,
so batching the whole test window into one call is both correct and much
faster than looping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import OHLC


@dataclass
class BacktestResult:
    model_name: str
    dates: pd.DatetimeIndex
    actual_close: np.ndarray
    predicted_close: np.ndarray
    mae: float
    rmse: float
    directional_accuracy: float
    metrics_all_fields: dict = field(default_factory=dict)


def train_test_split_by_date(df: pd.DataFrame, train_frac: float = 0.8):
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def run_backtest(model, df: pd.DataFrame, train_frac: float = 0.8) -> BacktestResult:
    train, test = train_test_split_by_date(df, train_frac)
    if len(test) < 2:
        raise ValueError("Not enough rows in the test window — need more history or a lower train_frac.")

    model.fit(train)

    # Forecast the WHOLE series in one vectorized call, then keep only the
    # predictions that land in the test window. Predictions for day t use
    # only true data from before day t, so restricting to the test window
    # afterwards introduces no leakage — it's equivalent to, but much faster
    # than, calling predict_next() once per test day.
    preds, dates = model.predict_series(df)
    dates = pd.DatetimeIndex(dates)

    test_start = test.index[0]
    mask = dates >= test_start
    pred_dates = dates[mask]
    pred_arr = preds[mask]

    if len(pred_dates) == 0:
        raise ValueError("No predictions fell inside the test window — check train_frac and history length.")

    actual_arr = df.loc[pred_dates, OHLC].to_numpy()

    close_idx = OHLC.index("Close")
    actual_close = actual_arr[:, close_idx]
    predicted_close = pred_arr[:, close_idx]

    mae = float(np.mean(np.abs(actual_close - predicted_close)))
    rmse = float(np.sqrt(np.mean((actual_close - predicted_close) ** 2)))

    # Directional accuracy: did the model get the up/down move right relative
    # to the previous day's actual close?
    # Note: the naive persistence model always predicts zero change, so its
    # predicted direction is "flat" (sign = 0). That structurally never
    # matches a real (non-zero) actual move, so naive persistence reports
    # ~0% here by construction -- not because the model is unusually bad,
    # but because "no predicted change" can't ever be "the right direction."
    # MAE/RMSE are the fairer comparison for this baseline.
    prev_close_full = df["Close"]
    prev_actual_close = prev_close_full.shift(1).loc[pred_dates].to_numpy()

    actual_dir = np.sign(actual_close - prev_actual_close)
    pred_dir = np.sign(predicted_close - prev_actual_close)
    directional_accuracy = float(np.mean(actual_dir == pred_dir))

    metrics_all_fields = {}
    for j, field_name in enumerate(OHLC):
        mae_j = float(np.mean(np.abs(actual_arr[:, j] - pred_arr[:, j])))
        rmse_j = float(np.sqrt(np.mean((actual_arr[:, j] - pred_arr[:, j]) ** 2)))
        metrics_all_fields[field_name] = {"mae": mae_j, "rmse": rmse_j}

    return BacktestResult(
        model_name=model.name,
        dates=pred_dates,
        actual_close=actual_close,
        predicted_close=predicted_close,
        mae=mae,
        rmse=rmse,
        directional_accuracy=directional_accuracy,
        metrics_all_fields=metrics_all_fields,
    )
