"""Run the exploratory workload analysis on the training window.

Usage::

    python -m cwfc.run_eda --data challenge_data --output-dir outputs/eda

Writes ``eda_summary.json``, CSV tables and (if matplotlib is installed) PNG
figures. Only the training window is analysed; holdout values are never
revealed and holdout readings are never loaded into any statistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import eda
from .splits import HORIZON, split_train_holdout
from .targets import (
    META_FILENAME,
    TARGET_CPU,
    clean_readings,
    load_cpu_readings,
    load_vm_metadata,
    targets_from_cleaned,
)

SEGMENT_COLUMNS = ("vm_category", "virtual_core_bucket", "memory_gb_bucket")


def run(
    data_dir: str | Path,
    horizon: int = HORIZON,
    output_dir: str | Path | None = None,
    figures: bool = True,
) -> dict:
    data_dir = Path(data_dir)
    cleaned = clean_readings(load_cpu_readings(data_dir))
    meta_path = data_dir / META_FILENAME
    meta = load_vm_metadata(meta_path) if meta_path.exists() else None
    series, _ = targets_from_cleaned(cleaned)

    split = split_train_holdout(series, horizon=horizon)
    train = split.train  # the only target data used below
    window_end = int(train.index.max())

    rows = eda.restrict_to_window(cleaned.rows, window_end)
    if int(rows["timestamp"].max()) > window_end:  # explicit leakage guard
        raise AssertionError("analysis rows extend beyond the training window")
    if train[TARGET_CPU].isna().any():
        raise ValueError("the training window contains gap timestamps; analyse segments separately first")

    x = eda.to_matrix(rows, "avg_cpu")
    mx = eda.to_matrix(rows, "max_cpu")
    y = train[TARGET_CPU].to_numpy(dtype=float)
    if not np.allclose(y, x.mean(axis=1).to_numpy()):
        raise AssertionError("target series and VM matrix disagree")

    peak = eda.peak_low_summary(y)
    summary: dict = {
        "coverage": eda.coverage_summary(train, rows),
        "reading_consistency": eda.reading_consistency(rows),
        "distribution": eda.distribution_summary(rows),
        "variance_structure": eda.variance_structure(x),
        "load_concentration": eda.load_concentration(x),
        "temporal": eda.temporal_summary(y),
        "peak_low": peak,
        "volatility_persistence": eda.volatility_persistence(x),
        "burstiness": eda.burstiness(rows),
    }
    tables: dict[str, pd.DataFrame] = {"candidate_targets": eda.candidate_targets(x, mx)}
    seg_series = None

    if meta is not None:
        summary["lifetimes"] = eda.lifetime_summary(meta, window_end)
        summary["tenants"] = eda.tenant_diversity(meta, x.columns)
        summary["metadata_representativeness"] = eda.metadata_representativeness(x, meta)
        episode = tuple(peak["peak_episode"]) if peak["peak_episode"] else None
        for col in SEGMENT_COLUMNS:
            if col in meta.columns:
                tables[f"segments_{col}"] = eda.segment_table(x, meta, col, episode)
        if "vm_category" in meta.columns:
            seg_series = eda.segment_series(x, meta, "vm_category")
            summary["category_correlation_levels"] = seg_series.corr().round(3).to_dict()
            summary["category_correlation_first_differences"] = seg_series.diff().dropna().corr().round(3).to_dict()

    paths: list[str] = []
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = dict(summary, candidate_targets=tables["candidate_targets"].to_dict("records"))
        (out / "eda_summary.json").write_text(json.dumps(payload, indent=2, default=str))
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        if figures:
            paths = _make_figures(out, y, peak, rows, x, summary, tables, seg_series)

    return {"summary": summary, "tables": tables, "figures": paths}


def _make_figures(out, y, peak, rows, x, summary, tables, seg_series) -> list[str]:
    try:
        from . import eda_plots as P
    except ImportError:  # matplotlib is an optional extra
        return []
    paths = [
        P.plot_aggregate(y, peak, out / "fig_aggregate_series.png"),
        P.plot_distribution(rows["avg_cpu"], x.mean(), out / "fig_cpu_distribution.png"),
        P.plot_acf(summary["temporal"]["acf"], summary["temporal"]["acf_95_band"], out / "fig_acf.png"),
        P.plot_load_concentration(x.mean(), out / "fig_load_concentration.png"),
        P.plot_candidate_targets(tables["candidate_targets"], out / "fig_candidate_targets.png"),
    ]
    if seg_series is not None and "segments_vm_category" in tables:
        paths.append(P.plot_segments(seg_series, tables["segments_vm_category"], out / "fig_segments_category.png"))
    return [str(p) for p in paths]


def findings(summary: dict) -> list[str]:
    """Short human-readable digest of the numbers (all from the training window)."""
    c, d, v, t, p = (summary[k] for k in ("coverage", "distribution", "variance_structure", "temporal", "peak_low"))
    vp = summary["volatility_persistence"]
    lines = [
        f"Window: {c['n_timestamps']} steps ({c['window_hours']:.1f} h), {c['n_vms']} VMs, "
        f"{c['n_vms_full_coverage']} with full coverage, {c['n_gap_timestamps']} gaps.",
        f"VM-level avg_cpu: mean {d['mean']:.2f}, median {d['quantiles']['p50']:.2f}, "
        f"p99 {d['quantiles']['p99']:.1f}, skew {d['skew']:.1f}; {100 * d['share_ge_80']:.2f}% of intervals >= 80%.",
        f"Variance: {100 * v['share_between_vm']:.0f}% between VMs, {100 * v['share_within_vm']:.0f}% within; "
        f"aggregate std {v['aggregate_std']:.3f} vs {v['aggregate_std_if_independent']:.3f} if independent "
        f"(~{v['effective_independent_vms']:.0f} effective independent VMs).",
        f"Aggregate: trend R2 {t['trend_r2']:.2f}, ACF1 {t['acf'][1]:.2f}, AR(1) phi {t['ar1_phi']:.2f}, "
        f"{100 * t['power_share_two_slowest_frequencies']:.0f}% of detrended power in the two slowest frequencies.",
        f"Peaks: p90 {p['peak_threshold']:.3f}, episode {p['peak_episode']}; peak is {p['peak_over_mean_pct']:.1f}% "
        f"above the mean, range {p['range_pct_of_mean']:.1f}% of the mean.",
        f"Out-of-sample cohort (top {vp['cohort_n_vms']} volatile VMs by history): overlap "
        f"{100 * vp['cohort_top_overlap_share']:.0f}% vs {100 * vp['cohort_overlap_if_random']:.0f}% by chance; "
        f"level persistence rho={vp['spearman_level']:.2f}.",
    ]
    if "lifetimes" in summary:
        lt = summary["lifetimes"]
        lines.append(
            f"Lifetimes: {100 * lt['share_created_at_trace_start']:.0f}% created at trace start, "
            f"{100 * lt['share_deleted_at_trace_end']:.0f}% alive at trace end; earliest deletion "
            f"{lt['seconds_from_window_end_to_earliest_deletion']} s after the window."
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    res = run(args.data, args.horizon, args.output_dir, figures=not args.no_figures)
    print("\n".join(findings(res["summary"])))
    if res["figures"]:
        print("figures:", *res["figures"], sep="\n  ")


if __name__ == "__main__":
    main()
