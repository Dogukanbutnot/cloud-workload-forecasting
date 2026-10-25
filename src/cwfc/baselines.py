from __future__ import annotations

import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)


def moving_average_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, window: int) -> pd.Series:
    """Flat forecast: mean of the last ``window`` observed values (NaN gaps ignored)."""
    if window <= 0:
        raise ValueError("window must be positive")
    if train.empty:
        raise ValueError("train must not be empty")
    tail = train[target].astype(float).iloc[-window:]
    if tail.notna().sum() == 0:
        raise ValueError("no observed values in the moving-average window")
    return pd.Series(float(tail.mean()), index=template.index, dtype=float)


def mean_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Flat forecast: mean of *all* history available at the forecast origin."""
    if train.empty:
        raise ValueError("train must not be empty")
    values = train[target].astype(float)
    if values.notna().sum() == 0:
        raise ValueError("no observed values in train")
    return pd.Series(float(values.mean()), index=template.index, dtype=float)
