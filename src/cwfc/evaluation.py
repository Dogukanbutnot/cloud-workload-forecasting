"""Rolling-origin evaluation of forecasters, with leakage-safe construction.

Leakage safeguards:

* A forecaster receives ``history`` (rows strictly before the origin) and a
  ``template`` that carries only the validation *timestamps*: it has no target
  columns, so the validation values cannot reach a forecaster even by accident.
* Every per-fold reference quantity is computed from the history only: the MASE
  scale (mean absolute one-step change) and the peak threshold (a quantile of
  the history).
* Callers pass only the *training window*. Nothing in this module can reach the
  holdout; see :class:`cwfc.splits.TrainHoldoutSplit`.

Metrics (``summarize``), all on the validation points of all folds:

* ``mae``, ``rmse``, ``smape`` (``cwfc.metrics``; sMAPE keeps ``eps`` in the
  denominator so a 0/0 pair scores 0 instead of producing NaN),
* ``bias``: mean signed error (forecast minus actual),
* ``mase``: per-fold MAE divided by the history's one-step naive MAE, averaged
  over folds; NaN when that scale is 0 (a constant series),
* ``mae_ratio_vs_persistence``: pooled MAE relative to persistence; NaN when the
  reference MAE is 0, because a ratio against a perfect forecast is undefined,
* ``folds_beating_persistence``: folds where the model's MAE is strictly lower,
* ``mae_fold_std``: spread of fold MAEs (folds overlap, so treat as optimistic),
* ``peak_mae`` / ``n_peak_points``: MAE where the actual value is at or above
  the ``peak_quantile`` of the history available at that origin.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .metrics import mae, rmse, smape
from .rolling import RollingFold

Forecaster = Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]
PEAK_QUANTILE = 0.9
REFERENCE_MODEL = "persistence"


def _naive_scale(values: np.ndarray) -> float:
    diffs = np.abs(np.diff(values))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    scale = float(diffs.mean())
    return scale if scale > 0 else float("nan")


def run_rolling_origin(
    train: pd.DataFrame,
    target: str,
    forecasters: Mapping[str, Forecaster],
    folds: list[RollingFold],
    peak_quantile: float = PEAK_QUANTILE,
) -> pd.DataFrame:
    """Forecast every fold with every forecaster; return a long prediction table.

    ``train`` must be the training window only. Columns of the result:
    ``fold, train_size, model, step, timestamp, y_true, y_pred, mase_scale,
    peak_threshold``.
    """
    if not folds:
        raise ValueError("at least one fold is required")
    if not 0 < peak_quantile < 1:
        raise ValueError("peak_quantile must be in (0, 1)")
    if max(f.val_stop for f in folds) > len(train):
        raise ValueError("folds extend beyond the supplied training window")

    frames = []
    for f in folds:
        history = train.iloc[: f.train_size]
        val = train.iloc[f.val_start : f.val_stop]
        template = val[[]]  # timestamps only: no target values reach forecasters
        hist_values = history[target].to_numpy(dtype=float)
        scale = _naive_scale(hist_values)
        threshold = float(np.nanquantile(hist_values, peak_quantile))

        for name, forecast in forecasters.items():
            pred = forecast(history.copy(), template.copy(), target)
            if not pred.index.equals(val.index):
                raise ValueError(f"{name}: forecast index does not match the validation timestamps")
            y_pred = pred.to_numpy(dtype=float)
            if not np.isfinite(y_pred).all():
                raise ValueError(
                    f"{name}: non-finite forecast at fold {f.fold}; handle gaps in "
                    "the history explicitly"
                )
            frames.append(
                pd.DataFrame(
                    {
                        "fold": f.fold,
                        "train_size": f.train_size,
                        "model": name,
                        "step": np.arange(1, len(val) + 1),
                        "timestamp": val.index.to_numpy(),
                        "y_true": val[target].to_numpy(dtype=float),
                        "y_pred": y_pred,
                        "mase_scale": scale,
                        "peak_threshold": threshold,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _fold_mae(df: pd.DataFrame) -> pd.Series:
    return (df["y_pred"] - df["y_true"]).abs().groupby(df["fold"]).mean()


def summarize(preds: pd.DataFrame, reference: str = REFERENCE_MODEL) -> pd.DataFrame:
    """One row per model, sorted by pooled MAE. See the module docstring."""
    scored = preds.dropna(subset=["y_true"])
    have_ref = reference in set(scored["model"])
    if have_ref:
        ref = scored[scored["model"] == reference]
        ref_mae = mae(ref["y_true"], ref["y_pred"])
        ref_fold_mae = _fold_mae(ref)

    rows = []
    for name, g in scored.groupby("model", sort=False):
        fold_mae = _fold_mae(g)
        fold_scale = g.groupby("fold")["mase_scale"].first()
        peak = g[g["y_true"] >= g["peak_threshold"]]
        m = mae(g["y_true"], g["y_pred"])
        row = {
            "model": name,
            "mae": m,
            "rmse": rmse(g["y_true"], g["y_pred"]),
            "smape_pct": smape(g["y_true"], g["y_pred"]),
            "bias": float((g["y_pred"] - g["y_true"]).mean()),
            "mase": float((fold_mae / fold_scale).mean()),
            "mae_fold_std": float(fold_mae.std(ddof=0)),
            "peak_mae": mae(peak["y_true"], peak["y_pred"]) if len(peak) else float("nan"),
            "n_peak_points": int(len(peak)),
            "n_folds": int(fold_mae.size),
        }
        if have_ref:
            row["mae_ratio_vs_persistence"] = m / ref_mae if ref_mae > 0 else float("nan")
            row["folds_beating_persistence"] = int((fold_mae < ref_fold_mae.reindex(fold_mae.index)).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae", kind="mergesort").reset_index(drop=True)


def error_by_horizon(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error at each forecast step (1..horizon), one column per model."""
    scored = preds.dropna(subset=["y_true"]).copy()
    scored["abs_error"] = (scored["y_pred"] - scored["y_true"]).abs()
    return scored.pivot_table(index="step", columns="model", values="abs_error", aggfunc="mean")


DEFAULT_BANDS = ((1, 6), (7, 12), (13, 24))


def band_mae(preds: pd.DataFrame, bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS) -> pd.DataFrame:
    """Mean absolute error per horizon band (inclusive step ranges), one column per model."""
    scored = preds.dropna(subset=["y_true"]).copy()
    scored["abs_error"] = (scored["y_pred"] - scored["y_true"]).abs()
    out = {}
    for lo, hi in bands:
        sel = scored[(scored["step"] >= lo) & (scored["step"] <= hi)]
        out[f"steps {lo}-{hi}"] = sel.groupby("model", sort=False)["abs_error"].mean()
    return pd.DataFrame(out).T


def fold_mae_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-fold MAE, one column per model (rows are fold ids)."""
    scored = preds.dropna(subset=["y_true"]).copy()
    scored["abs_error"] = (scored["y_pred"] - scored["y_true"]).abs()
    return scored.pivot_table(index="fold", columns="model", values="abs_error", aggfunc="mean")


def baseline_envelope(by_horizon: pd.DataFrame, baseline_names: list[str]) -> pd.Series:
    """Best baseline error at each step.

    The best baseline is picked per step on the same folds that are scored, so
    this is an *optimistic* reference: beating it is a strong result, falling a
    little short of it is not evidence of a weak model.
    """
    return by_horizon[baseline_names].min(axis=1).rename("best_baseline_per_step")


def regret_summary(fold_mae: pd.DataFrame, models: list[str], references: list[str]) -> pd.DataFrame:
    """Per-fold regret of each model against the best reference in that fold.

    Regret is ``MAE(model) - min over references of MAE`` in the same fold, so a
    negative value means the model beat every reference there. With a series
    whose dynamics change inside the window (see the exploratory analysis) no
    single baseline wins every fold; a model that is never far behind the best
    one is the robust choice, and ``max_regret`` measures exactly that.
    """
    best = fold_mae[references].min(axis=1)
    rows = []
    for m in models:
        regret = fold_mae[m] - best
        rows.append(
            {
                "model": m,
                "mean_regret": float(regret.mean()),
                "max_regret": float(regret.max()),
                "folds_beating_all_references": int((regret < 0).sum()),
                "n_folds": int(len(regret)),
            }
        )
    return pd.DataFrame(rows).sort_values("max_regret", kind="mergesort").reset_index(drop=True)
