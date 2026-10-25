"""The ``predictions.csv`` contract from SUBMISSION.md, as executable checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def contract_problems(df: pd.DataFrame, expected_timestamps) -> list[str]:
    """Return every violation of the submission contract (empty list = valid)."""
    expected = np.asarray(list(expected_timestamps))
    problems: list[str] = []

    if list(df.columns) != COLUMNS:
        problems.append(f"columns must be exactly {COLUMNS} in that order, got {list(df.columns)}")
        return problems  # nothing else can be checked reliably

    if len(df) != len(expected):
        problems.append(f"expected {len(expected)} rows, got {len(df)}")
    if df["timestamp"].duplicated().any():
        problems.append("duplicate timestamps")

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy()
    if len(df) == len(expected) and not np.array_equal(ts, expected):
        problems.append("timestamps do not match the holdout timestamps exactly, in chronological order")
    elif len(df) != len(expected) and not set(ts.tolist()) <= set(expected.tolist()):
        problems.append("timestamps contain values outside the holdout")

    for col in COLUMNS[1:]:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            problems.append(f"{col} contains missing or non-finite values")
    active = pd.to_numeric(df["active_vm_count"], errors="coerce")
    cpu = pd.to_numeric(df["aggregate_cpu_utilization"], errors="coerce")
    if (active < 0).any():
        problems.append("active_vm_count must be >= 0")
    if ((cpu < 0) | (cpu > 100)).any():
        problems.append("aggregate_cpu_utilization must be in [0, 100]")
    return problems


def validate_submission(df: pd.DataFrame, expected_timestamps) -> None:
    """Raise ``ValueError`` listing all contract violations."""
    problems = contract_problems(df, expected_timestamps)
    if problems:
        raise ValueError("invalid submission: " + "; ".join(problems))


def write_submission(df: pd.DataFrame, path: str | Path, expected_timestamps) -> Path:
    """Validate, then write without an index column."""
    validate_submission(df, expected_timestamps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)
    return path
