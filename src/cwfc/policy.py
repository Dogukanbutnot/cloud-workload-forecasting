"""A minimal capacity-headroom policy built from rolling-origin residuals.

Policy: provision ``forecast + margin(h)`` where ``margin(h)`` is a high
quantile of how much the actual value exceeded the forecast in the training-
window folds, estimated separately per horizon band (errors grow with ``h``).

Trade-off (ASSIGNMENT.md, section G):

* a **small** margin wastes little capacity but is exceeded more often
  (under-provisioning: the shortfall is a service risk);
* a **large** margin almost never falls short but reserves capacity that stays
  idle (over-provisioning: the surplus is a cost).

``under_rate`` and ``mean_surplus`` measure the two sides. The margin is
estimated from training-window folds only, and the quantile is fixed in advance
(not tuned on the holdout). Fold residuals overlap and there are few of them,
so the estimated margin carries real uncertainty; the trade-off table computed
on the same folds is in-sample and therefore optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import DEFAULT_BANDS


def margin_by_band(
    preds: pd.DataFrame,
    model: str,
    quantile: float = 0.9,
    bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS,
) -> dict[tuple[int, int], float]:
    """Per-band margin: the ``quantile`` of ``actual - forecast``, floored at 0."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    sel = preds[(preds["model"] == model)].dropna(subset=["y_true"])
    if sel.empty:
        raise ValueError(f"no scored predictions for model {model!r}")
    shortfall = sel["y_true"] - sel["y_pred"]
    out = {}
    for lo, hi in bands:
        in_band = shortfall[(sel["step"] >= lo) & (sel["step"] <= hi)]
        out[(lo, hi)] = max(float(in_band.quantile(quantile)), 0.0) if len(in_band) else float("nan")
    return out


def margins_for_steps(steps, margins: dict[tuple[int, int], float]) -> np.ndarray:
    """Look up the margin of each forecast step (steps beyond the last band use it)."""
    steps = np.asarray(steps)
    ordered = sorted(margins.items())
    out = np.empty(len(steps), dtype=float)
    for i, s in enumerate(steps):
        chosen = ordered[-1][1]
        for (lo, hi), m in ordered:
            if lo <= s <= hi:
                chosen = m
                break
        out[i] = chosen
    return out


def evaluate_policy(y_true, y_pred, margin) -> dict:
    """Under-/over-provisioning outcome of capacity ``y_pred + margin``."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    capacity = y_pred + np.asarray(margin, float)
    short = y_true > capacity
    return {
        "mean_margin": float(np.mean(margin)),
        "under_rate": float(short.mean()),
        "mean_shortfall_when_under": float((y_true - capacity)[short].mean()) if short.any() else 0.0,
        "mean_surplus": float(np.clip(capacity - y_true, 0, None).mean()),
        "n": int(len(y_true)),
    }


def tradeoff_table(
    preds: pd.DataFrame,
    model: str,
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS,
) -> pd.DataFrame:
    """In-sample (training-fold) trade-off for several margin quantiles."""
    sel = preds[preds["model"] == model].dropna(subset=["y_true"])
    rows = []
    for q in quantiles:
        m = margins_for_steps(sel["step"], margin_by_band(preds, model, q, bands))
        rows.append({"quantile": q, **evaluate_policy(sel["y_true"], sel["y_pred"], m)})
    return pd.DataFrame(rows)
