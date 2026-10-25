"""Exploratory workload analysis (ASSIGNMENT.md, section 3A).

Everything here operates on the **training window** only. Callers restrict the
cleaned readings with :func:`restrict_to_window` and never pass holdout rows;
``cwfc.run_eda`` enforces and tests this. Static VM metadata (categories,
lifetimes) is not a target value and is used freely.

Each function answers one question and returns plain numbers or a DataFrame, so
the report and the notebook quote exactly what the code computed. Statistics
that are *selected on the data they describe* say so in their docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import LIFETIME_METADATA_COLUMNS
from .targets import STEP_SECONDS

# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #


def restrict_to_window(rows: pd.DataFrame, window_end: int) -> pd.DataFrame:
    """Keep readings with ``timestamp <= window_end`` (the training window)."""
    return rows[rows["timestamp"] <= window_end]


def to_matrix(rows: pd.DataFrame, value: str = "avg_cpu") -> pd.DataFrame:
    """Timestamp x VM matrix of one reading column."""
    return rows.pivot(index="timestamp", columns="vm_id", values=value)


def _f(x) -> float:
    return float(x)


def _corr(a, b, method: str = "pearson") -> float:
    """Correlation that is NaN (without warnings) when either input is constant."""
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    if a.std() == 0 or b.std() == 0 or len(a) < 2:
        return float("nan")
    if method == "spearman":  # Pearson correlation of average ranks; avoids pandas' optional scipy dependency
        a, b = a.rank(), b.rank()
    elif method != "pearson":
        raise ValueError(f"unsupported correlation method {method!r}")
    return _f(a.corr(b))


# --------------------------------------------------------------------------- #
# 1. Cadence and coverage; 2. missingness and irregularities
# --------------------------------------------------------------------------- #


def coverage_summary(series_window: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """Cadence, window length and per-VM coverage of the training window."""
    idx = series_window.index.to_numpy()
    steps = np.unique(np.diff(idx)) if len(idx) > 1 else np.array([], dtype=int)
    per_vm = rows.groupby("vm_id").size()
    return {
        "n_timestamps": int(len(idx)),
        "window_start": int(idx[0]),
        "window_end": int(idx[-1]),
        "window_hours": _f(len(idx) * STEP_SECONDS / 3600),
        "observed_step_seconds": [int(s) for s in steps],
        "n_gap_timestamps": int(series_window["is_gap"].sum()),
        "n_rows": int(len(rows)),
        "n_vms": int(len(per_vm)),
        "readings_per_vm_min": int(per_vm.min()),
        "readings_per_vm_max": int(per_vm.max()),
        "n_vms_full_coverage": int((per_vm == len(idx)).sum()),
    }


def reading_consistency(rows: pd.DataFrame) -> dict:
    """Sanity checks on min/avg/max within a reading, plus saturation."""
    eps = 1e-9
    return {
        "n_avg_outside_min_max": int(
            ((rows["avg_cpu"] < rows["min_cpu"] - eps) | (rows["avg_cpu"] > rows["max_cpu"] + eps)).sum()
        ),
        "share_avg_exactly_zero": _f((rows["avg_cpu"] == 0).mean()),
        "share_max_cpu_ge_95": _f((rows["max_cpu"] >= 95).mean()),
        "share_max_cpu_eq_100": _f((rows["max_cpu"] >= 100).mean()),
    }


# --------------------------------------------------------------------------- #
# 3. VM lifetimes and churn
# --------------------------------------------------------------------------- #


def lifetime_summary(meta: pd.DataFrame, window_end: int, trace_end: int | None = None) -> dict:
    """Censoring structure of VM lifetimes relative to the challenge window.

    ``trace_end`` defaults to the latest deletion timestamp (an *inferred* end
    of the source trace). VMs deleted exactly then are right-censored.
    """
    created = meta["timestamp_vm_created"]
    deleted = meta["timestamp_vm_deleted"]
    trace_start = int(created.min())
    trace_end = int(deleted.max()) if trace_end is None else int(trace_end)
    early = deleted < trace_end
    lifetimes_days = ((deleted - created)[early] / 86400.0)
    out = {
        "n_vms": int(len(meta)),
        "trace_start": trace_start,
        "trace_end_inferred": trace_end,
        "share_created_at_trace_start": _f((created == trace_start).mean()),
        "share_deleted_at_trace_end": _f((~early).mean()),
        "n_deleted_before_trace_end": int(early.sum()),
        "n_deleted_within_window": int((deleted <= window_end).sum()),
        "earliest_deletion": int(deleted.min()),
        "seconds_from_window_end_to_earliest_deletion": int(deleted.min() - window_end),
    }
    if len(lifetimes_days):
        q = lifetimes_days.quantile([0.1, 0.5, 0.9])
        out["early_deleted_lifetime_days_p10_p50_p90"] = [_f(v) for v in q]
    return out


# --------------------------------------------------------------------------- #
# 4. CPU utilisation distributions and variance structure
# --------------------------------------------------------------------------- #


def distribution_summary(rows: pd.DataFrame) -> dict:
    """Pooled VM-level ``avg_cpu`` distribution (all VM-intervals)."""
    v = rows["avg_cpu"]
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quant = {f"p{int(round(q * 100))}": _f(v.quantile(q)) for q in qs}
    return {
        "mean": _f(v.mean()),
        "std": _f(v.std()),
        "skew": _f(v.skew()),
        "mean_over_median": _f(v.mean() / v.median()),
        "quantiles": quant,
        "share_lt_1": _f((v < 1).mean()),
        "share_lt_5": _f((v < 5).mean()),
        "share_ge_50": _f((v >= 50).mean()),
        "share_ge_80": _f((v >= 80).mean()),
    }


def variance_structure(x: pd.DataFrame, top_fracs: tuple[float, ...] = (0.001, 0.01, 0.1)) -> dict:
    """Between/within-VM variance, diversification and concentration.

    * ``share_between_vm``: part of pooled variance explained by VMs having
      different average levels (vs. varying over time).
    * ``variance_ratio_vs_independent``: variance of the aggregate divided by
      what it would be if VMs fluctuated independently. 1 means perfectly
      diversified; large means shared (common) movement.
    * ``mean_pairwise_corr`` / ``effective_independent_vms``: equicorrelation
      approximation (a summary, not a claim that all pairs are equal).
    * ``top_share_in_sample``: share of aggregate variance attributed to the
      VMs with the largest covariance with the aggregate. **Selected on the same
      data it describes, so it overstates how predictable the concentration
      is**; see :func:`volatility_persistence` for the out-of-sample version.
    """
    n_vms = x.shape[1]
    agg = x.mean(axis=1)
    vm_mean = x.mean()
    vm_var0 = x.var(ddof=0)
    between, within = _f(vm_mean.var(ddof=0)), _f(vm_var0.mean())

    sigma2 = x.var(ddof=1)
    var_agg = _f(agg.var(ddof=1))
    var_indep = _f(sigma2.sum() / n_vms**2)
    avg_cov = (var_agg * n_vms**2 - _f(sigma2.sum())) / (n_vms * (n_vms - 1))
    rho = _f(avg_cov / (np.sqrt(sigma2).mean() ** 2))
    n_eff = _f(1.0 / (rho + (1 - rho) / n_vms)) if rho > 0 else float(n_vms)

    xc, ac = x - vm_mean, agg - agg.mean()
    contrib = (xc.mul(ac, axis=0).sum() / (len(agg) - 1)) / (n_vms * var_agg)
    ranked = contrib.sort_values(ascending=False)
    top = {}
    for frac in top_fracs:
        k = max(1, int(np.ceil(frac * n_vms)))
        top[f"top_{frac * 100:g}pct"] = {"n_vms": k, "variance_share": _f(ranked.iloc[:k].sum())}

    return {
        "n_vms": int(n_vms),
        "per_vm_mean_quantiles": {f"p{int(q * 100)}": _f(vm_mean.quantile(q)) for q in (0.5, 0.9, 0.99)},
        "per_vm_temporal_std_median": _f(x.std().median()),
        "per_vm_temporal_std_mean": _f(x.std().mean()),
        "share_between_vm": between / (between + within),
        "share_within_vm": within / (between + within),
        "aggregate_std": _f(np.sqrt(var_agg)),
        "aggregate_std_if_independent": _f(np.sqrt(var_indep)),
        "variance_ratio_vs_independent": var_agg / var_indep,
        "mean_pairwise_corr": rho,
        "effective_independent_vms": n_eff,
        "top_share_in_sample": top,
        "n_vms_negative_contribution": int((contrib < 0).sum()),
    }


def load_concentration(x: pd.DataFrame, fracs: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)) -> dict:
    """Share of total CPU load carried by the busiest VMs (by window mean).

    This is about *where the load is*, not where the variance is; both matter for
    capacity planning, and the first is known at any origin from past readings.
    """
    vm_mean = x.mean().sort_values(ascending=False)
    total = vm_mean.sum()
    out = {}
    for frac in fracs:
        k = max(1, int(np.ceil(frac * len(vm_mean))))
        out[f"top_{frac * 100:g}pct"] = {"n_vms": k, "load_share": _f(vm_mean.iloc[:k].sum() / total)}
    return out


# --------------------------------------------------------------------------- #
# 5. Temporal patterns
# --------------------------------------------------------------------------- #


def temporal_summary(y, max_lag: int = 48) -> dict:
    """Trend, autocorrelation, spectrum and mean reversion of a 1-D series.

    The periodogram only resolves periods up to the window length, so with ~110
    five-minute points nothing about a daily cycle can be concluded; this
    function reports how much power sits in the slowest frequencies instead.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(n)
    constant = bool(np.std(y) == 0)  # every dynamic statistic below is undefined for a constant series
    slope, intercept = np.polyfit(t, y, 1)
    r2 = 0.0 if constant else _f(np.corrcoef(t, y)[0, 1] ** 2)

    yc = y - y.mean()
    denom = (yc**2).sum()
    lags = min(max_lag, n - 1)
    if constant:
        acf = np.full(lags + 1, np.nan)
    else:
        acf = np.array([(yc[: n - k] * yc[k:]).sum() / denom for k in range(lags + 1)])
    below = np.flatnonzero(acf < 0.2)

    detr = y - (slope * t + intercept)
    power = np.abs(np.fft.rfft(detr)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    order = np.argsort(power[1:])[::-1][:3] + 1
    total = power[1:].sum()
    # Residual power of a perfectly linear series is floating-point noise, not signal.
    has_signal = (not constant) and total > 1e-20 * n * float(np.var(y))

    def share(p: float) -> float:
        return _f(p / total) if has_signal else 0.0

    low = share(power[1:3].sum())  # the two slowest non-zero frequencies

    phi = float("nan") if constant else _f(np.polyfit(y[:-1], y[1:], 1)[0])
    d = np.diff(y)
    return {
        "n": int(n),
        "mean": _f(y.mean()),
        "std": _f(y.std(ddof=1)),
        "trend_per_step": _f(slope),
        "trend_over_window": _f(slope * n),
        "trend_r2": r2,
        "acf": [_f(a) for a in acf],
        "acf_95_band": _f(1.96 / np.sqrt(n)),
        "first_lag_acf_below_0_2": int(below[0]) if len(below) else None,
        "acf1_of_first_difference": (
            _f(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 2 and np.std(d) > 0 else None
        ),
        "periodogram_top_periods_steps": [_f(1.0 / freqs[i]) for i in order],
        "periodogram_top_power_share": [share(power[i]) for i in order],
        "power_share_two_slowest_frequencies": low,
        "ar1_phi": phi,
        "ar1_half_life_steps": _f(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else None,
    }


def _runs(mask: np.ndarray) -> list[dict]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append({"start": int(i), "end": int(j), "length": int(j - i + 1)})
            i = j + 1
        else:
            i += 1
    return out


def peak_low_summary(y, hi_q: float = 0.9, lo_q: float = 0.1) -> dict:
    """Peak/low-demand points and contiguous episodes within the window.

    Thresholds are quantiles of the window itself, matching the peak definition
    used in validation. ``peak_episode`` is the span from the first to the last
    peak point (descriptive; it does not claim the span is contiguous).
    """
    y = np.asarray(y, dtype=float)
    hi, lo = float(np.quantile(y, hi_q)), float(np.quantile(y, lo_q))
    peak_runs, low_runs = _runs(y >= hi), _runs(y <= lo)
    peak_idx = np.flatnonzero(y >= hi)
    mean = y.mean()
    return {
        "peak_threshold": hi,
        "low_threshold": lo,
        "peak_runs": peak_runs,
        "low_runs": low_runs,
        "n_peak_points": int(len(peak_idx)),
        "peak_episode": [int(peak_idx.min()), int(peak_idx.max())] if len(peak_idx) else None,
        "n_low_runs_of_length_1": int(sum(r["length"] == 1 for r in low_runs)),
        "argmax": int(y.argmax()),
        "argmin": int(y.argmin()),
        "peak_over_mean_pct": _f(100 * (y.max() - mean) / mean),
        "trough_under_mean_pct": _f(100 * (mean - y.min()) / mean),
        "range_pct_of_mean": _f(100 * (y.max() - y.min()) / mean),
    }


# --------------------------------------------------------------------------- #
# 6. Segments
# --------------------------------------------------------------------------- #


def segment_series(x: pd.DataFrame, meta: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean ``avg_cpu`` per timestamp for each level of a metadata column."""
    m = meta.drop_duplicates("vm_id").set_index("vm_id").reindex(x.columns)
    out = {}
    for level, ids in m.groupby(column).groups.items():
        out[str(level)] = x[list(ids)].mean(axis=1)
    return pd.DataFrame(out)


def segment_table(
    x: pd.DataFrame, meta: pd.DataFrame, column: str, episode: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Level, spread and dynamics of each segment of a metadata column.

    ``episode`` (positions, inclusive) marks a window span of interest, e.g.
    the peak episode; ``episode_amp_pp`` is the segment's mean inside it minus
    its mean outside it. ``trend_pp_over_window`` is the fitted linear change of
    the segment's mean over the window and ``trend_contribution_to_aggregate_pp``
    that change weighted by the segment's share of VMs.
    """
    n_vms, agg_mean = x.shape[1], x.mean(axis=1).mean()
    vm_mean, vm_std = x.mean(), x.std()
    m = meta.drop_duplicates("vm_id").set_index("vm_id").reindex(x.columns)
    mask = None
    if episode is not None:
        mask = np.zeros(len(x), dtype=bool)
        mask[episode[0] : episode[1] + 1] = True

    rows = []
    for level, ids in m.groupby(column).groups.items():
        ids = list(ids)
        s = x[ids].mean(axis=1)
        row = {
            "segment": str(level),
            "n_vms": len(ids),
            "share_vms_pct": 100 * len(ids) / n_vms,
            "mean_cpu": _f(s.mean()),
            "median_vm_mean": _f(vm_mean[ids].median()),
            "p95_vm_mean": _f(vm_mean[ids].quantile(0.95)),
            "mean_vm_temporal_std": _f(vm_std[ids].mean()),
            "segment_series_std": _f(s.std()),
            "share_of_level_pct": 100 * (len(ids) / n_vms) * _f(s.mean()) / _f(agg_mean),
            "acf1": _f(pd.Series(s.to_numpy()).autocorr(1)),
        }
        slope = np.polyfit(np.arange(len(s)), s.to_numpy(), 1)[0]
        row["trend_pp_over_window"] = _f(slope * len(s))
        row["trend_contribution_to_aggregate_pp"] = _f(slope * len(s) * len(ids) / n_vms)
        if mask is not None and mask.any() and (~mask).any():
            row["episode_amp_pp"] = _f(s.to_numpy()[mask].mean() - s.to_numpy()[~mask].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_vms", ascending=False).reset_index(drop=True)


def tenant_diversity(meta: pd.DataFrame, vm_ids) -> dict:
    """How concentrated are the VMs across subscriptions and deployments?"""
    m = meta[meta["vm_id"].isin(set(vm_ids))]
    n = len(m)
    out = {}
    for col, name in (("subscription_id", "subscriptions"), ("deployment_id", "deployments")):
        if col in m:
            counts = m[col].value_counts()
            out[name] = {
                "n": int(len(counts)),
                "largest_share_pct": _f(100 * counts.iloc[0] / n),
                "top10_share_pct": _f(100 * counts.iloc[:10].sum() / n),
            }
    return out


# --------------------------------------------------------------------------- #
# Predictability of the structure found above
# --------------------------------------------------------------------------- #


def volatility_persistence(x: pd.DataFrame, split_at: int | None = None, cohort_frac: float = 0.05) -> dict:
    """Are VM levels and volatility stable enough to be known at an origin?

    The window is cut at ``split_at`` (default: the middle). Per-VM mean and
    temporal std from the first part are compared with those of the second
    (both parts lie inside the training window). The "cohort" is the
    ``cohort_frac`` most volatile VMs *according to the first part only*; this
    is the honest, out-of-sample counterpart of the in-sample concentration in
    :func:`variance_structure`.
    """
    n, n_vms = len(x), x.shape[1]
    h = n // 2 if split_at is None else split_at
    if h < 3 or n - h < 3:
        raise ValueError("each part needs at least 3 observations")
    s1, s2 = x.iloc[:h].std(), x.iloc[h:].std()
    k = max(1, int(round(cohort_frac * n_vms)))
    top1, top2 = set(s1.nlargest(k).index), set(s2.nlargest(k).index)
    cohort = s1.nlargest(k).index
    rest = x.columns.difference(cohort)
    fut = x.iloc[h:]
    part_c = fut[cohort].sum(axis=1) / n_vms
    part_r = fut[rest].sum(axis=1) / n_vms
    total = fut.mean(axis=1)
    return {
        "split_at": int(h),
        "spearman_level": _corr(x.iloc[:h].mean(), fut.mean(), "spearman"),
        "spearman_volatility": _corr(s1, s2, "spearman"),
        "cohort_n_vms": int(k),
        "cohort_top_overlap_share": _f(len(top1 & top2) / k),
        "cohort_overlap_if_random": _f(k / n_vms),
        "future_std_total": _f(total.std()),
        "future_std_cohort_part": _f(part_c.std()),
        "future_std_rest_part": _f(part_r.std()),
        "cohort_share_of_level_pct": _f(100 * part_c.mean() / total.mean()),
        "corr_total_cohort_part": _corr(total, part_c),
        "corr_total_rest_part": _corr(total, part_r),
    }


# --------------------------------------------------------------------------- #
# Operational framing: alternative targets, burstiness, metadata
# --------------------------------------------------------------------------- #


def candidate_targets(avg: pd.DataFrame, mx: pd.DataFrame) -> pd.DataFrame:
    """Compare the required target with operationally motivated alternatives.

    ``cv_pct`` is std/mean and ``range_pct_of_mean`` is (max-min)/mean, both over
    the window: a target with a tiny CV hardly moves, so errors on it are easy
    to mistake for skill or noise.
    """
    cands = {
        "mean_avg_cpu (required)": avg.mean(axis=1),
        "p95_avg_cpu_across_vms": avg.quantile(0.95, axis=1),
        "p99_avg_cpu_across_vms": avg.quantile(0.99, axis=1),
        "share_vms_avg_cpu_ge_50_pct": (avg >= 50).mean(axis=1) * 100,
        "share_vms_avg_cpu_ge_80_pct": (avg >= 80).mean(axis=1) * 100,
        "mean_max_cpu_across_vms": mx.mean(axis=1),
        "share_vms_max_cpu_ge_95_pct": (mx >= 95).mean(axis=1) * 100,
    }
    rows = []
    for name, s in cands.items():
        rows.append(
            {
                "target": name,
                "mean": _f(s.mean()),
                "std": _f(s.std()),
                "cv_pct": _f(100 * s.std() / s.mean()) if s.mean() != 0 else float("nan"),
                "acf1": _f(s.autocorr(1)) if s.std() > 0 else float("nan"),
                "range_pct_of_mean": (
                    _f(100 * (s.max() - s.min()) / s.mean()) if s.mean() != 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def burstiness(rows: pd.DataFrame) -> dict:
    """How much do five-minute averages hide short spikes?"""
    gap = rows["max_cpu"] - rows["avg_cpu"]
    return {
        "mean_max_minus_avg_pp": _f(gap.mean()),
        "median_max_minus_avg_pp": _f(gap.median()),
        "share_intervals_max_ge_95": _f((rows["max_cpu"] >= 95).mean()),
        "share_intervals_hidden_spike": _f(((rows["avg_cpu"] < 20) & (rows["max_cpu"] >= 90)).mean()),
        "n_intervals_hidden_spike": int(((rows["avg_cpu"] < 20) & (rows["max_cpu"] >= 90)).sum()),
    }


def metadata_representativeness(x: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """How well does the window reflect the lifetime statistics in metadata?

    Descriptive only: these lifetime columns contain post-origin information and
    must not be model inputs (``cwfc.availability``).
    """
    m = meta.drop_duplicates("vm_id").set_index("vm_id").reindex(x.columns)
    vm_mean = x.mean()
    out = {
        "window_aggregate_mean": _f(x.mean(axis=1).mean()),
        "lifetime_columns_present": sorted(LIFETIME_METADATA_COLUMNS & set(meta.columns)),
    }
    if "avg_cpu" in m:
        out.update(
            lifetime_avg_cpu_mean=_f(m["avg_cpu"].mean()),
            pearson_window_vs_lifetime=_corr(vm_mean, m["avg_cpu"]),
            spearman_window_vs_lifetime=_corr(vm_mean, m["avg_cpu"], "spearman"),
        )
    return out
