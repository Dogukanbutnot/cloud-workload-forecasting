"""Mean-reverting level model for the aggregate CPU series.

Motivation (exploratory analysis, training window only): the aggregate has no
trend and no identifiable cycle, a fast AR(1)-like short-term dependence
(phi ~ 0.67) and, on top, a slow component. Rolling-origin baselines showed that
persistence wins at short horizons and the history mean at long horizons, with
a crossover around step 8-9. This model interpolates between the two::

    level_t   = alpha * y_t + (1 - alpha) * level_{t-1}        (smoothed "now")
    mean_t    = mean(y_0 .. y_t)                                (long-run level)
    forecast_h = mean_t + phi**h * (level_t - mean_t),  h = 1, 2, ...

* ``alpha = 1`` gives damped persistence toward the mean (an AR(1) forecast);
  ``phi -> 1`` gives persistence (of the smoothed level); ``phi -> 0`` gives the
  history mean. All the baselines are limits of this family.
* Estimation. ``(alpha, phi)`` are chosen on a small grid by minimising the mean
  absolute multi-step error *inside the history it is given*: for every origin
  ``s`` in the history and every ``h <= horizon`` with ``s + h`` still inside
  the history, the forecast built from ``y_0 .. y_s`` alone is compared with
  ``y_{s+h}``. The same ``mean_s`` / ``level_s`` that would be available at
  that origin are used, so the inner loss is itself leakage-free. Because the
  fit sees only the history passed in, running it inside each rolling-origin
  fold is a properly nested procedure: no fold validation value ever influences
  its own parameters.
* Ties in the grid are broken deterministically (first in grid order).

The model needs no features and no tuning data beyond the history, which suits a
110-point training window where anything with more freedom mostly fits noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

ALPHAS = tuple(np.round(np.arange(0.1, 1.0001, 0.1), 2))
PHIS = tuple(np.round(np.r_[np.arange(0.5, 0.9501, 0.05), 0.97, 0.99], 2))
MIN_ORIGIN_FLOOR = 10


@dataclass(frozen=True)
class MeanRevertingLevelFit:
    alpha: float
    phi: float
    level: float  # smoothed level at the last observation
    mean: float  # mean of all observations
    inner_loss: float  # mean absolute multi-step error in the history
    n_obs: int

    def forecast(self, steps: int) -> np.ndarray:
        h = np.arange(1, steps + 1)
        return self.mean + self.phi**h * (self.level - self.mean)


def ses_levels(y: np.ndarray, alpha: float) -> np.ndarray:
    """Simple exponential smoothing, initialised at the first observation."""
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for t in range(1, len(y)):
        out[t] = alpha * y[t] + (1.0 - alpha) * out[t - 1]
    return out


def fit_mean_reverting_level(
    y,
    horizon: int,
    alphas: tuple[float, ...] = ALPHAS,
    phis: tuple[float, ...] = PHIS,
    min_origin: int | None = None,
) -> MeanRevertingLevelFit:
    """Estimate ``(alpha, phi)`` from the history ``y`` and return the fit."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not np.isfinite(y).all():
        raise ValueError("history contains NaN/inf; handle gaps before fitting")
    start = max(MIN_ORIGIN_FLOOR, n // 4) if min_origin is None else min_origin
    if n < start + 2:
        raise ValueError(f"history of {n} points is too short to estimate the model (need >= {start + 2})")

    running_mean = np.cumsum(y) / np.arange(1, n + 1)
    max_h = min(horizon, n - 1 - start)
    best: tuple[float, float, float] | None = None  # (loss, alpha, phi)
    for a in alphas:
        dev = ses_levels(y, a) - running_mean  # level - mean, per origin
        for p in phis:
            total, count = 0.0, 0
            for h in range(1, max_h + 1):
                origins = np.arange(start, n - h)
                err = np.abs(y[origins + h] - (running_mean[origins] + p**h * dev[origins]))
                total += float(err.sum())
                count += len(origins)
            loss = total / count
            if best is None or loss < best[0] - 1e-15:
                best = (loss, float(a), float(p))

    loss, a, p = best
    level = float(ses_levels(y, a)[-1])
    return MeanRevertingLevelFit(a, p, level, float(running_mean[-1]), loss, n)


def mean_reverting_level_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    alphas: tuple[float, ...] = ALPHAS,
    phis: tuple[float, ...] = PHIS,
) -> pd.Series:
    """Forecaster with the same signature as the baselines in ``cwfc.baselines``.

    The forecast horizon is the length of ``template`` (timestamps only).
    """
    fit = fit_mean_reverting_level(train[target].to_numpy(dtype=float), len(template), alphas, phis)
    return pd.Series(fit.forecast(len(template)), index=template.index, dtype=float)


# A damped-persistence ablation: no smoothing, only the reversion speed is fitted.
damped_persistence_forecast = partial(mean_reverting_level_forecast, alphas=(1.0,))


def model_suite() -> dict:
    return {
        "mean_reverting_level": mean_reverting_level_forecast,
        "damped_persistence": damped_persistence_forecast,
    }
