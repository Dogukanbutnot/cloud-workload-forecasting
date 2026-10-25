"""One-shot evaluation of ``predictions.csv`` on the public holdout.

Usage (only after the model is frozen)::

    python -m cwfc.evaluate_holdout --data challenge_data --predictions predictions.csv --confirm

This is the **only** module that calls ``reveal_holdout()`` (``tests/`` enforces
it by scanning the source tree). Running it is a one-way door: anything changed
in the modelling after looking at these numbers would break the protocol in
ASSIGNMENT.md. The evaluated object is the submitted CSV itself, not a forecast
recomputed here, so what is scored is what is delivered. Reference forecasts
(persistence, history mean) are built from the training window only.

Reported per target: MAE, RMSE, sMAPE (``cwfc.metrics`` keeps ``eps`` in the
denominator so near-zero pairs do not blow up), bias, MASE (scale = mean
absolute one-step change in the *training* window), MAE ratio versus
persistence and history mean (NaN when the reference is exact), MAE by horizon
band, MAE on peak points (actual >= the training-window p90) and the largest
errors. The validation-fold MAE range is included so the holdout number can be
read against what rolling-origin validation led us to expect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import mean_forecast, persistence_forecast
from .evaluation import DEFAULT_BANDS
from .metrics import mae, rmse, smape
from .policy import evaluate_policy, margin_by_band, margins_for_steps, tradeoff_table
from .run_baselines import MIN_TRAIN_SIZE, STEP
from .splits import HORIZON, split_train_holdout
from .submission import validate_submission
from .targets import TARGET_ACTIVE, TARGET_CPU, clean_readings, load_cpu_readings, targets_from_cleaned

PEAK_QUANTILE = 0.9
POLICY_QUANTILE = 0.9  # fixed in advance, never tuned on the holdout


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def _naive_scale(values: np.ndarray) -> float:
    d = np.abs(np.diff(values[np.isfinite(values)]))
    s = float(d.mean()) if d.size else float("nan")
    return s if s > 0 else float("nan")


def _metrics(y: np.ndarray, f: np.ndarray, scale: float) -> dict:
    m = mae(y, f)
    return {
        "mae": m,
        "rmse": rmse(y, f),
        "smape_pct": smape(y, f),
        "bias": float(np.mean(f - y)),
        "mase": float(m / scale) if np.isfinite(scale) else float("nan"),
    }


def run(
    data_dir: str | Path,
    predictions_path: str | Path,
    horizon: int = HORIZON,
    output_dir: str | Path | None = None,
    min_train_size: int = MIN_TRAIN_SIZE,
    step: int = STEP,
) -> dict:
    from . import run_model  # local import: only needed for validation-fold context

    series, _ = targets_from_cleaned(clean_readings(load_cpu_readings(Path(data_dir))))
    split = split_train_holdout(series, horizon=horizon)
    sub = pd.read_csv(predictions_path)
    validate_submission(sub, split.holdout_timestamps)

    truth = split.reveal_holdout()  # <- the single place holdout values are read
    train = split.train
    template = pd.DataFrame(index=split.holdout_timestamps)
    steps = np.arange(1, len(truth) + 1)
    sub = sub.set_index("timestamp").reindex(truth.index)

    context = run_model.run(data_dir, horizon, min_train_size, step)
    fold_mae = context["fold_mae"]["damped_persistence"]

    result: dict = {
        "n_holdout": int(len(truth)),
        "validation_fold_mae_damped_persistence": {
            "min": float(fold_mae.min()),
            "mean": float(fold_mae.mean()),
            "max": float(fold_mae.max()),
        },
        "targets": {},
    }
    per_step = pd.DataFrame({"step": steps}, index=truth.index)

    for target in (TARGET_ACTIVE, TARGET_CPU):
        y = truth[target].to_numpy(dtype=float)
        scale = _naive_scale(train[target].to_numpy(dtype=float))
        forecasts = {
            "submission": sub[target].to_numpy(dtype=float),
            "persistence": persistence_forecast(train, template, target).to_numpy(dtype=float),
            "history_mean": mean_forecast(train, template, target).to_numpy(dtype=float),
        }
        block = {name: _metrics(y, f, scale) for name, f in forecasts.items()}
        sub_mae = block["submission"]["mae"]
        block["submission"]["mae_ratio_vs_persistence"] = _ratio(sub_mae, block["persistence"]["mae"])
        block["submission"]["mae_ratio_vs_history_mean"] = _ratio(sub_mae, block["history_mean"]["mae"])

        extras: dict = {"actual_mean": float(y.mean()), "actual_min": float(y.min()), "actual_max": float(y.max())}
        err = forecasts["submission"] - y
        bands = {}
        for lo, hi in DEFAULT_BANDS:
            sel = (steps >= lo) & (steps <= hi)
            if sel.any():  # a short horizon (smoke data) has no steps in the later bands
                bands[f"steps {lo}-{hi}"] = {name: float(np.abs(f - y)[sel].mean()) for name, f in forecasts.items()}
        extras["mae_by_band"] = bands
        top = np.argsort(-np.abs(err))[:3]
        extras["largest_errors"] = [
            {"timestamp": int(truth.index[i]), "step": int(steps[i]), "actual": float(y[i]),
             "forecast": float(forecasts["submission"][i]), "error": float(err[i])}
            for i in top
        ]

        if target == TARGET_CPU:
            thr = float(np.quantile(train[target].to_numpy(dtype=float), PEAK_QUANTILE))
            peak = y >= thr
            extras["peak_threshold_from_training_p90"] = thr
            extras["n_peak_points"] = int(peak.sum())
            if peak.any():
                extras["peak_mae"] = {name: float(np.abs(f - y)[peak].mean()) for name, f in forecasts.items()}
            top_q = y >= np.quantile(y, 0.75)  # descriptive: the highest quarter of the holdout itself
            extras["top_quartile_of_holdout_mae"] = {
                name: float(np.abs(f - y)[top_q].mean()) for name, f in forecasts.items()
            }

            margins = margin_by_band(context["predictions"], "damped_persistence", POLICY_QUANTILE)
            m = margins_for_steps(steps, margins)
            extras["policy"] = {
                "quantile": POLICY_QUANTILE,
                "margin_by_band": {f"steps {lo}-{hi}": v for (lo, hi), v in margins.items()},
                "holdout_with_margin": evaluate_policy(y, forecasts["submission"], m),
                "holdout_without_margin": evaluate_policy(y, forecasts["submission"], np.zeros_like(m)),
                "training_folds_tradeoff": tradeoff_table(context["predictions"], "damped_persistence").to_dict("records"),
            }
            per_step["capacity_with_margin"] = forecasts["submission"] + m
        block["details"] = extras
        result["targets"][target] = block
        per_step[f"{target}_actual"] = y
        per_step[f"{target}_forecast"] = forecasts["submission"]

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "holdout_metrics.json").write_text(json.dumps(result, indent=2, default=float))
        per_step.to_csv(out / "holdout_forecast_vs_actual.csv")
        try:
            from .report_plots import plot_holdout

            plot_holdout(train[TARGET_CPU], truth[TARGET_CPU], sub[TARGET_CPU], per_step["capacity_with_margin"],
                         {k: v for k, v in {"persistence": persistence_forecast(train, template, TARGET_CPU),
                                            "history_mean": mean_forecast(train, template, TARGET_CPU)}.items()},
                         out / "fig_holdout.png")
        except ImportError:
            pass
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--predictions", default="predictions.csv")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--min-train-size", type=int, default=MIN_TRAIN_SIZE)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--confirm", action="store_true", help="acknowledge that this reveals the holdout (run once, after freezing the model)")
    args = parser.parse_args(argv)
    if not args.confirm:
        raise SystemExit("Refusing to reveal the holdout without --confirm. Freeze the model first.")
    result = run(args.data, args.predictions, args.horizon, args.output_dir, args.min_train_size, args.step)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
