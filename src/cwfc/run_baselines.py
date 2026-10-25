"""Evaluate baselines with rolling-origin validation on the training window.

Usage::

    python -m cwfc.run_baselines --data challenge_data [--output-dir outputs]

Only the first ``len(series) - horizon`` observations (the training window) are
used. The holdout values are never revealed here.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd

from .baselines import (
    mean_forecast,
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
)
from .evaluation import Forecaster, error_by_horizon, run_rolling_origin, summarize
from .population import check_constant_population, fixed_population_forecast
from .rolling import rolling_origin_folds
from .splits import HORIZON, split_train_holdout
from .targets import (
    META_FILENAME,
    TARGET_ACTIVE,
    TARGET_CPU,
    clean_readings,
    load_cpu_readings,
    load_vm_metadata,
    targets_from_cleaned,
)

MIN_TRAIN_SIZE = 50
STEP = 6
MA_WINDOWS = (3, 6, 12, 24)
SEASON_LENGTHS = (12, 24)  # 1 h and 2 h; a daily season (288) cannot be tested in 110 points


def baseline_suite() -> dict[str, Forecaster]:
    suite: dict[str, Forecaster] = {
        "persistence": persistence_forecast,
        "history_mean": mean_forecast,
    }
    for w in MA_WINDOWS:
        suite[f"moving_average_{w}"] = partial(moving_average_forecast, window=w)
    for s in SEASON_LENGTHS:
        suite[f"seasonal_naive_{s}"] = partial(seasonal_naive_forecast, season_length=s)
    return suite


def fixed_population_forecaster(cleaned_rows: pd.DataFrame, meta: pd.DataFrame) -> Forecaster:
    """Re-prove the fixed population at *each* origin from its own history."""

    def forecast(history: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
        report = check_constant_population(
            history, cleaned_rows, meta, horizon_end=int(template.index.max())
        )
        return fixed_population_forecast(report, template.index)

    return forecast


def run(
    data_dir: str | Path,
    horizon: int = HORIZON,
    min_train_size: int = MIN_TRAIN_SIZE,
    step: int = STEP,
    output_dir: str | Path | None = None,
) -> dict:
    data_dir = Path(data_dir)
    cleaned = clean_readings(load_cpu_readings(data_dir))
    meta_path = data_dir / META_FILENAME
    meta = load_vm_metadata(data_dir) if meta_path.exists() else None
    series, _ = targets_from_cleaned(cleaned)

    split = split_train_holdout(series, horizon=horizon)
    train = split.train  # the only data this module ever touches
    folds = rolling_origin_folds(len(train), horizon, min_train_size, step)

    population = check_constant_population(
        train, cleaned.rows, meta, horizon_end=int(split.holdout_timestamps.max())
    )

    result: dict = {"folds": folds, "population": population, "targets": {}}
    for target in (TARGET_ACTIVE, TARGET_CPU):
        suite = baseline_suite()
        if target == TARGET_ACTIVE and population.fixed_by_construction:
            suite["fixed_population"] = fixed_population_forecaster(cleaned.rows, meta)
        preds = run_rolling_origin(train, target, suite, folds)
        result["targets"][target] = {
            "predictions": preds,
            "summary": summarize(preds),
            "by_horizon": error_by_horizon(preds),
        }
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out / f"rolling_predictions_{target}.csv", index=False)
            result["targets"][target]["summary"].to_csv(out / f"baseline_summary_{target}.csv", index=False)
            result["targets"][target]["by_horizon"].to_csv(out / f"baseline_error_by_horizon_{target}.csv")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--min-train-size", type=int, default=MIN_TRAIN_SIZE)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    res = run(args.data, args.horizon, args.min_train_size, args.step, args.output_dir)
    folds = res["folds"]
    print(
        f"{len(folds)} rolling-origin folds, horizon {folds[0].horizon}: train sizes "
        f"{[f.train_size for f in folds]}; last validation window ends at position {folds[-1].val_stop}"
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for target, parts in res["targets"].items():
        print(f"\n=== {target} ===")
        print(parts["summary"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
