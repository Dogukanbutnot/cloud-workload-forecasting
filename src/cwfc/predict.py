"""Produce ``predictions.csv`` from the supplied release data.

Usage::

    python -m cwfc.predict --data challenge_data --output predictions.csv

Method
------
1. Build the full target series, then hold out the final ``horizon`` (24)
   observations. Only the training window (everything before them) is used
   below; holdout *values* are never read here (``tests/test_predict.py``
   enforces this by making ``reveal_holdout`` raise).
2. ``aggregate_cpu_utilization``: damped persistence toward the history mean,
   ``mean + phi**h * (last - mean)``, with ``phi`` estimated from the whole
   training window (``cwfc.models``). The choice of this model was made on
   rolling-origin validation inside the training window (see the report).
3. ``active_vm_count``: the constant population, but only when it is *proven*
   fixed (constant on the training window, full VM coverage, and no VM created
   after the window start or deleted before the horizon end, per the metadata;
   see ``cwfc.population``). Otherwise the last observed count is carried
   forward, so the pipeline does not silently assume a constant on data where
   it does not hold (for example a reviewer's smoke set).
4. Fallbacks, reported in the summary: a training window shorter than
   ``MIN_HISTORY`` points uses persistence for the CPU target as well.
5. Values are clipped to the contract ranges and validated before writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baselines import persistence_forecast
from .models import fit_mean_reverting_level
from .population import check_constant_population, fixed_population_forecast
from .splits import HORIZON, split_train_holdout
from .submission import COLUMNS, write_submission
from .targets import (
    META_FILENAME,
    TARGET_ACTIVE,
    TARGET_CPU,
    clean_readings,
    load_cpu_readings,
    load_vm_metadata,
    targets_from_cleaned,
)

MIN_HISTORY = 12  # below this the model cannot be estimated meaningfully


def predict(data_dir: str | Path, horizon: int = HORIZON) -> tuple[pd.DataFrame, dict]:
    """Return the submission frame and a JSON-serialisable summary."""
    data_dir = Path(data_dir)
    cleaned = clean_readings(load_cpu_readings(data_dir))
    # --data may be the release directory or the CPU readings file itself
    meta_path = (data_dir if data_dir.is_dir() else data_dir.parent) / META_FILENAME
    meta = load_vm_metadata(meta_path) if meta_path.exists() else None
    series, _ = targets_from_cleaned(cleaned)

    split = split_train_holdout(series, horizon=horizon)  # holdout values stay hidden
    train = split.train
    index = split.holdout_timestamps
    template = pd.DataFrame(index=index)
    summary: dict = {"train_size": split.train_size, "horizon": split.horizon}

    # --- aggregate_cpu_utilization --------------------------------------- #
    if train[TARGET_CPU].isna().any():
        raise ValueError(
            "the training window has gap timestamps, so the CPU series cannot be fitted without "
            "inventing values; inspect the data release"
        )
    if split.train_size >= MIN_HISTORY:
        fit = fit_mean_reverting_level(train[TARGET_CPU].to_numpy(dtype=float), horizon, alphas=(1.0,))
        cpu = pd.Series(fit.forecast(horizon), index=index, dtype=float)
        summary["cpu_method"] = "damped_persistence"
        summary["cpu_phi"] = fit.phi
        summary["cpu_history_mean"] = fit.mean
        summary["cpu_last_value"] = fit.level
    else:
        cpu = persistence_forecast(train, template, TARGET_CPU)
        summary["cpu_method"] = "persistence_fallback_short_history"

    # --- active_vm_count -------------------------------------------------- #
    population = None
    if meta is not None:
        population = check_constant_population(
            train, cleaned.rows, meta, horizon_end=int(index.max())
        )
        summary["population_verdict"] = population.verdict
    if population is not None and population.fixed_by_construction:
        active = fixed_population_forecast(population, index)
        summary["active_method"] = "fixed_population"
    else:
        active = persistence_forecast(train, template, TARGET_ACTIVE).astype(float)
        summary["active_method"] = "persistence_fallback_population_not_proven_fixed"

    out = pd.DataFrame(
        {
            "timestamp": index.to_numpy(),
            TARGET_ACTIVE: active.clip(lower=0).to_numpy(),
            TARGET_CPU: cpu.clip(lower=0.0, upper=100.0).to_numpy(),
        }
    )[COLUMNS]
    summary["holdout_timestamps"] = [int(index.min()), int(index.max())]
    return out, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data", help="data directory with the release files")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args(argv)

    frame, summary = predict(args.data, args.horizon)
    # re-derive the expected timestamps from the data so the validator is independent of predict()
    series, _ = targets_from_cleaned(clean_readings(load_cpu_readings(Path(args.data))))
    expected = split_train_holdout(series, horizon=args.horizon).holdout_timestamps
    path = write_submission(frame, args.output, expected)
    print(json.dumps({"written": str(path), "rows": len(frame), **summary}, indent=2, default=float))


if __name__ == "__main__":
    main()
