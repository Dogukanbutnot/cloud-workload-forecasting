"""Build the target series, apply the 110/24 split, run the population check.

Usage::

    python -m cwfc.build_targets --data challenge_data [--output-dir outputs]

The report printed here contains training-window statistics and holdout
*timestamps* only. Holdout target values are never printed or written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .population import check_constant_population
from .splits import HORIZON, is_official_protocol, split_train_holdout
from .targets import (
    META_FILENAME,
    TARGET_CPU,
    clean_readings,
    load_cpu_readings,
    load_vm_metadata,
    targets_from_cleaned,
)


def run(data_dir: str | Path, horizon: int = HORIZON, output_dir: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    cpu = load_cpu_readings(data_dir)
    meta_path = data_dir / META_FILENAME
    meta = load_vm_metadata(data_dir) if meta_path.exists() else None

    cleaned = clean_readings(cpu)
    series, build_report = targets_from_cleaned(cleaned)
    split = split_train_holdout(series, horizon=horizon)

    population = check_constant_population(
        split.train,
        cleaned.rows,
        meta,
        horizon_end=int(split.holdout_timestamps.max()),
    )

    train_util = split.train[TARGET_CPU].dropna()
    summary = {
        "build": build_report.to_dict(),
        "split": {
            "train_size": split.train_size,
            "horizon": split.horizon,
            "train_start": int(split.train.index.min()),
            "train_end": int(split.train.index.max()),
            "holdout_start": int(split.holdout_timestamps.min()),
            "holdout_end": int(split.holdout_timestamps.max()),
            "official_110_24": is_official_protocol(split),
        },
        "population": population.to_dict(),
        "train_aggregate_cpu_utilization": {
            "mean": float(train_util.mean()),
            "std": float(train_util.std()),
            "min": float(train_util.min()),
            "max": float(train_util.max()),
        },
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        split.train.to_csv(out / "targets_train.csv")
        split.holdout_timestamps.to_series().to_csv(out / "holdout_timestamps.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data", help="data directory or CPU readings file")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--output-dir", default=None, help="optional: write train targets and holdout timestamps")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data, args.horizon, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
