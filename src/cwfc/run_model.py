"""Evaluate the main model against the baselines, rolling-origin, training window only.

Usage::

    python -m cwfc.run_model --data challenge_data [--output-dir outputs/model]

``active_vm_count`` is handled by the proven fixed-population forecast (see
``cwfc.population``); this module evaluates ``aggregate_cpu_utilization``. The
holdout values are never revealed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    band_mae,
    baseline_envelope,
    error_by_horizon,
    fold_mae_table,
    regret_summary,
    run_rolling_origin,
    summarize,
)
from .models import fit_mean_reverting_level, model_suite
from .rolling import rolling_origin_folds
from .run_baselines import MIN_TRAIN_SIZE, STEP, baseline_suite
from .splits import HORIZON, split_train_holdout
from .targets import TARGET_CPU, clean_readings, load_cpu_readings, targets_from_cleaned


def run(
    data_dir: str | Path,
    horizon: int = HORIZON,
    min_train_size: int = MIN_TRAIN_SIZE,
    step: int = STEP,
    output_dir: str | Path | None = None,
) -> dict:
    series, _ = targets_from_cleaned(clean_readings(load_cpu_readings(Path(data_dir))))
    split = split_train_holdout(series, horizon=horizon)
    train = split.train  # the only data this module touches
    folds = rolling_origin_folds(len(train), horizon, min_train_size, step)

    baselines = baseline_suite()
    suite = {**baselines, **model_suite()}
    preds = run_rolling_origin(train, TARGET_CPU, suite, folds)
    by_horizon = error_by_horizon(preds)
    by_horizon["best_baseline_per_step"] = baseline_envelope(by_horizon, list(baselines))

    params = pd.DataFrame(
        [
            {
                "fold": f.fold,
                "train_size": f.train_size,
                **{
                    k: getattr(fit, k)
                    for k in ("alpha", "phi", "level", "mean", "inner_loss")
                },
            }
            for f in folds
            for fit in [fit_mean_reverting_level(train[TARGET_CPU].to_numpy()[: f.train_size], f.horizon)]
        ]
    )

    result = {
        "folds": folds,
        "predictions": preds,
        "summary": summarize(preds),
        "bands": band_mae(preds),
        "by_horizon": by_horizon,
        "fold_mae": fold_mae_table(preds),
        "params": params,
    }
    result["regret"] = regret_summary(result["fold_mae"], list(suite), list(baselines))
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out / "rolling_predictions.csv", index=False)
        result["summary"].to_csv(out / "model_summary.csv", index=False)
        result["bands"].to_csv(out / "model_error_by_band.csv")
        by_horizon.to_csv(out / "model_error_by_horizon.csv")
        result["fold_mae"].to_csv(out / "model_fold_mae.csv")
        params.to_csv(out / "model_parameters_by_fold.csv", index=False)
        result["regret"].to_csv(out / "model_regret_vs_best_baseline.csv", index=False)
        try:
            from .report_plots import plot_model_comparison

            plot_model_comparison(
                result["bands"], result["fold_mae"], [f.train_size for f in folds], horizon,
                out / "fig_model_comparison.png",
            )
        except ImportError:  # matplotlib is an optional extra
            pass
    return result


def paired_vs(fold_mae: pd.DataFrame, model: str, other: str) -> dict:
    """Fold-level comparison: model minus other MAE (negative favours the model)."""
    d = fold_mae[model] - fold_mae[other]
    return {
        "mean_diff": float(d.mean()),
        "std_diff": float(d.std(ddof=0)),
        "folds_model_better": int((d < 0).sum()),
        "n_folds": int(len(d)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--min-train-size", type=int, default=MIN_TRAIN_SIZE)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    res = run(args.data, args.horizon, args.min_train_size, args.step, args.output_dir)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = ["model", "mae", "rmse", "bias", "mase", "peak_mae", "mae_ratio_vs_persistence", "folds_beating_persistence"]
    print(f"{len(res['folds'])} folds; aggregate_cpu_utilization, training window only\n")
    print(res["summary"][show].round(4).to_string(index=False))
    print("\nMAE by horizon band")
    cols = ["persistence", "history_mean", "mean_reverting_level", "damped_persistence"]
    print(res["bands"][cols].round(4).to_string())
    print("\nfitted parameters per fold")
    print(res["params"].round(3).to_string(index=False))
    print("\nregret vs the best BASELINE in each fold (negative = beat all baselines in that fold)")
    print(res["regret"].round(4).to_string(index=False))
    for other in ("persistence", "history_mean"):
        print(f"\nmean_reverting_level vs {other}:", paired_vs(res["fold_mae"], "mean_reverting_level", other))


if __name__ == "__main__":
    main()
