from __future__ import annotations

from typing import Callable

import pandas as pd

from cwfc.split import HOLDOUT_SIZE
from cwfc.validation import rolling_origin_splits

MIN_TRAIN_SIZE = 50
STEP = 6

Forecaster = Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]


def backtest(
    train: pd.DataFrame,
    forecaster: Forecaster,
    target: str,
    horizon: int = HOLDOUT_SIZE,
    min_train_size: int = MIN_TRAIN_SIZE,
    step: int = STEP,
) -> pd.DataFrame:
    """Rolling-origin backtest over the training window.

    Returns one row per (fold, horizon step) with the actual and forecast values.
    """
    # rolling_origin_splits works on datetimes; the raw timestamp is seconds since start.
    frame = train.assign(ts=pd.to_datetime(train["timestamp"], unit="s", utc=True))
    splits = rolling_origin_splits(
        frame, timestamp_col="ts", horizon=horizon, min_train_size=min_train_size, step=step
    )
    if not splits:
        raise ValueError(
            f"No folds: {len(train)} rows is too short for min_train_size={min_train_size} "
            f"and horizon={horizon}."
        )

    rows = []
    for fold, split in enumerate(splits):
        fit = frame[frame["ts"] <= split.train_end]
        validation = frame[(frame["ts"] >= split.validation_start) & (frame["ts"] <= split.validation_end)]
        forecast = forecaster(fit, validation, target)
        rows.append(pd.DataFrame({
            "fold": fold,
            "step": range(1, len(validation) + 1),
            "timestamp": validation["timestamp"].to_numpy(),
            "actual": validation[target].to_numpy(),
            "forecast": forecast.to_numpy(),
        }))

    return pd.concat(rows, ignore_index=True)
