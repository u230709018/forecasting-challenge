from __future__ import annotations

import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)

def mean_forecast(
    train: pd.DataFrame, template: pd.DataFrame, target: str, window: int | None = None
) -> pd.Series:
    if window is None:
        value = train[target].mean()
    else:
        value = train[target].iloc[-window:].mean()
    return pd.Series(value, index=template.index, dtype=float)


def decay_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    t_mean = train[target].mean()
    last_val = train[target].iloc[-1]
    phi = train[target].autocorr(lag=1)

    if pd.isna(phi) or phi < 0:
        phi = 0
        
    guesses = [ t_mean + (last_val - t_mean) * phi**h for h in range(1, len(template) + 1) ]
    
    return pd.Series(guesses, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
