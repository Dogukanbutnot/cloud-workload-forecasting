"""Figures for the exploratory analysis. Requires ``matplotlib`` (``pip install .[eda]``).

Each function saves one PNG and returns its path. Time axes are hours since the
start of the window, because the source timestamps are relative to the trace.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .targets import STEP_SECONDS  # noqa: E402

DPI = 130
COLORS = {"main": "#1f4e79", "accent": "#c0504d", "muted": "#8c8c8c", "band": "#f2c14e"}


def _hours(n: int) -> np.ndarray:
    return np.arange(n) * STEP_SECONDS / 3600.0


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_aggregate(y: np.ndarray, peak: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    h = _hours(len(y))
    ax.plot(h, y, color=COLORS["main"], lw=1.6, label="aggregate mean avg_cpu")
    ax.axhline(y.mean(), color=COLORS["muted"], lw=1, ls="--", label="window mean")
    ax.axhline(peak["peak_threshold"], color=COLORS["accent"], lw=1, ls=":", label="p90 (peak threshold)")
    ax.axhline(peak["low_threshold"], color=COLORS["muted"], lw=1, ls=":", label="p10 (low threshold)")
    if peak["peak_episode"]:
        a, b = peak["peak_episode"]
        ax.axvspan(h[a], h[b], color=COLORS["band"], alpha=0.3, label="peak episode")
    ax.set_xlabel("hours since window start")
    ax.set_ylabel("CPU utilisation (%), y-axis truncated")
    ax.set_title("Aggregate CPU utilisation, training window")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    return _save(fig, path)


def plot_distribution(values: pd.Series, vm_means: pd.Series, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(values, bins=100, range=(0, 100), color=COLORS["main"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("VM-interval avg_cpu (%)")
    axes[0].set_ylabel("count (log scale)")
    axes[0].set_title("Pooled VM-level readings: long right tail")
    v = np.sort(vm_means.to_numpy())
    axes[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS["main"])
    for q, c in ((0.5, COLORS["muted"]), (0.95, COLORS["accent"])):
        val = float(np.quantile(v, q))
        axes[1].axvline(val, color=c, ls="--", lw=1)
        axes[1].text(val + 1, 0.05 + 0.1 * (q > 0.9), f"p{int(q * 100)}={val:.1f}", fontsize=8, color=c)
    axes[1].set_xlabel("per-VM window mean avg_cpu (%)")
    axes[1].set_ylabel("cumulative share of VMs")
    axes[1].set_title("VM levels (empirical CDF)")
    return _save(fig, path)


def plot_acf(acf: list[float], band: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    lags = np.arange(len(acf))
    ax.bar(lags[1:], np.asarray(acf)[1:], color=COLORS["main"], width=0.8)
    ax.axhspan(-band, band, color=COLORS["muted"], alpha=0.25, label="95% band under white noise")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("lag (5-minute steps)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("ACF of the aggregate: fast decay, no recurring cycle")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_segments(seg: pd.DataFrame, table: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    h = _hours(len(seg))
    for col in seg.columns:
        axes[0].plot(h, seg[col] - seg[col].mean(), lw=1.3, label=col)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xlabel("hours since window start")
    axes[0].set_ylabel("deviation from segment mean (pp)")
    axes[0].set_title("Segment aggregates (de-meaned)")
    axes[0].legend(fontsize=7)
    t = table.set_index("segment")
    axes[1].bar(t.index, t["mean_cpu"], color=COLORS["main"])
    axes[1].set_ylabel("mean CPU (%)")
    axes[1].set_title("Level by segment")
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
    return _save(fig, path)


def plot_load_concentration(vm_means: pd.Series, path: Path) -> Path:
    v = np.sort(vm_means.to_numpy())[::-1]
    cum = np.cumsum(v) / v.sum()
    frac = np.arange(1, len(v) + 1) / len(v)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(frac * 100, cum * 100, color=COLORS["main"], label="busiest VMs first")
    ax.plot([0, 100], [0, 100], color=COLORS["muted"], ls="--", label="equal load")
    for p in (1, 10):
        i = int(np.ceil(p / 100 * len(v))) - 1
        ax.scatter([p], [cum[i] * 100], color=COLORS["accent"], zorder=3)
        ax.annotate(f"top {p}% -> {cum[i] * 100:.0f}% of load", (p, cum[i] * 100), textcoords="offset points",
                    xytext=(8, -14), fontsize=8)
    ax.set_xlabel("share of VMs (%)")
    ax.set_ylabel("share of total CPU load (%)")
    ax.set_title("CPU load is concentrated in few VMs")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_candidate_targets(table: pd.DataFrame, path: Path) -> Path:
    t = table.set_index("target")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["cv_pct"], height=0.4, color=COLORS["main"], label="CV (std/mean, %)")
    ax.barh(y + 0.2, t["range_pct_of_mean"], height=0.4, color=COLORS["band"], label="range / mean (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("variability within the training window (%)")
    ax.set_title("How much does each candidate target actually move?")
    ax.legend(fontsize=8)
    return _save(fig, path)
