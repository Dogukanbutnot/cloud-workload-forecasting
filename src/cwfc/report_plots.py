"""Figures for the modelling results. Requires ``matplotlib`` (``pip install .[eda]``)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {
    "persistence": ("#8c8c8c", "o"),
    "history_mean": ("#c0504d", "s"),
    "damped_persistence": ("#1f4e79", "D"),
}


def plot_model_comparison(bands: pd.DataFrame, fold_mae: pd.DataFrame, train_sizes: list[int], horizon: int, path: Path) -> Path:
    """Left: MAE by horizon band. Right: MAE of each fold (validation window shown on the x axis)."""
    models = list(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})

    x = np.arange(len(bands))
    w = 0.26
    for i, m in enumerate(models):
        axes[0].bar(x + (i - 1) * w, bands[m], width=w, color=STYLE[m][0], label=m)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bands.index, fontsize=8)
    axes[0].set_ylabel("MAE (pp), training-window folds")
    axes[0].set_title("Error by horizon band")
    axes[0].legend(fontsize=7)

    labels = [f"{s}-{s + horizon - 1}" for s in train_sizes]
    xs = np.arange(len(fold_mae))
    for m in models:
        axes[1].plot(xs, fold_mae[m], marker=STYLE[m][1], color=STYLE[m][0], label=m, lw=1.2)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_xlabel("validation window (positions in the training window)")
    axes[1].set_ylabel("MAE (pp)")
    axes[1].set_title("Error by fold: no single baseline wins everywhere")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_holdout(train: pd.Series, truth: pd.Series, forecast: pd.Series, capacity: pd.Series,
                 baselines: dict[str, pd.Series], path: Path, tail: int = 48) -> Path:
    """Last ``tail`` training points, the holdout actuals, the submitted forecast and the headroom line."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    tr = train.iloc[-tail:]
    h0 = tr.index[0]

    def hours(idx):
        return (np.asarray(idx) - h0) / 3600.0

    ax.plot(hours(tr.index), tr.to_numpy(), color="#8c8c8c", lw=1.4, label="training window (tail)")
    ax.plot(hours(truth.index), truth.to_numpy(), color="black", lw=1.6, label="holdout actual")
    ax.plot(hours(forecast.index), forecast.to_numpy(), color="#1f4e79", lw=2, label="submission (damped persistence)")
    for name, s in baselines.items():
        ax.plot(hours(s.index), s.to_numpy(), lw=1, ls=":", label=name,
                color={"persistence": "#c0504d", "history_mean": "#9bbb59"}.get(name, "#777777"))
    ax.plot(hours(capacity.index), capacity.to_numpy(), color="#f2994a", lw=1.2, ls="--", label="forecast + headroom")
    ax.axvline(hours([truth.index[0]])[0] - 150 / 3600, color="black", lw=0.6)
    ax.set_xlabel("hours since the start of the plotted window")
    ax.set_ylabel("CPU utilisation (%), y-axis truncated")
    ax.set_title("Holdout: submitted forecast against what happened")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
