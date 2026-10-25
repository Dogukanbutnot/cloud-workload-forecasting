"""Construction of the two required aggregate target series.

Targets (DATASET.md, section 4), per timestamp ``t``:

* ``active_vm_count(t)``: number of distinct VM ids with a *valid* CPU
  observation at ``t``.
* ``aggregate_cpu_utilization(t)``: mean of the VM-level ``avg_cpu`` values at
  ``t``, in percent, in ``[0, 100]``.

Documented treatment of irregularities (every choice is reported, never silent):

* **Valid observation.** A row is valid iff its timestamp, ``vm_id`` and
  ``avg_cpu`` are present, finite, and ``avg_cpu`` lies in ``[0, 100]``.
  Invalid rows are dropped *before* de-duplication so that a bad duplicate can
  never shadow a good one. Both counts are reported.
* **Duplicates.** If a ``(vm_id, timestamp)`` pair appears more than once, the
  first occurrence (in input order) is kept so every VM is counted at most
  once per timestamp. The number of dropped rows and the number of keys whose
  duplicates disagreed on ``avg_cpu`` are reported.
* **Grid.** Source timestamps are treated as a 300-second grid anchored at the
  earliest timestamp. Off-grid timestamps raise ``ValueError``: we refuse to
  resample silently. No resampling is performed on grid-aligned data, so the
  required five-minute cadence is preserved and no mapping back is needed.
* **Gaps.** A grid timestamp with zero valid observations is kept in the
  series and flagged ``is_gap``. Its ``active_vm_count`` is 0 (that is the true
  count), while ``aggregate_cpu_utilization`` is NaN. We never interpolate or
  forward-fill a target value (DATASET.md, section 5).

Nothing here looks at, selects on, or depends on the train/holdout split: the
full series is built first and split afterwards (ASSIGNMENT.md, section 3B).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STEP_SECONDS = 300
CPU_COLUMNS = ("timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu")
TARGET_ACTIVE = "active_vm_count"
TARGET_CPU = "aggregate_cpu_utilization"
TARGETS = (TARGET_ACTIVE, TARGET_CPU)
CPU_MIN, CPU_MAX = 0.0, 100.0
CPU_FILENAME = "vm_cpu_readings.csv.gz"
META_FILENAME = "vm_metadata.csv.gz"


@dataclass(frozen=True)
class CleanedReadings:
    """Validated, de-duplicated VM-level rows plus an audit trail."""

    rows: pd.DataFrame
    n_rows_in: int
    n_invalid_rows_dropped: int
    n_duplicate_rows_dropped: int
    n_conflicting_duplicate_keys: int
    raw_ts_min: int
    raw_ts_max: int
    step_seconds: int


@dataclass(frozen=True)
class TargetBuildReport:
    """Audit trail of how the target series was built (JSON-serialisable)."""

    n_rows_in: int
    n_invalid_rows_dropped: int
    n_duplicate_rows_dropped: int
    n_conflicting_duplicate_keys: int
    n_rows_used: int
    n_vms: int
    grid_start: int
    grid_end: int
    step_seconds: int
    n_grid_points: int
    n_gap_timestamps: int
    gap_timestamps: tuple[int, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gap_timestamps"] = list(self.gap_timestamps)
        return d


def _resolve(path: str | Path, filename: str) -> Path:
    p = Path(path)
    return p / filename if p.is_dir() else p


def load_cpu_readings(path: str | Path) -> pd.DataFrame:
    """Load VM-level CPU readings from a file, or from a data directory."""
    return pd.read_csv(_resolve(path, CPU_FILENAME))


def load_vm_metadata(path: str | Path) -> pd.DataFrame:
    """Load VM metadata from a file, or from a data directory."""
    return pd.read_csv(_resolve(path, META_FILENAME))


def clean_readings(
    cpu: pd.DataFrame, step_seconds: int = STEP_SECONDS
) -> CleanedReadings:
    """Validate and de-duplicate raw readings. See the module docstring."""
    missing = [c for c in CPU_COLUMNS if c not in cpu.columns]
    if missing:
        raise ValueError(f"CPU readings are missing required columns: {missing}")

    n_in = len(cpu)
    ts = pd.to_numeric(cpu["timestamp"], errors="coerce")
    avg = pd.to_numeric(cpu["avg_cpu"], errors="coerce")

    ts_ok = np.isfinite(ts)
    if not ts_ok.any():
        raise ValueError("No finite timestamps found in the CPU readings.")
    finite_ts = ts[ts_ok]
    if (finite_ts % 1 != 0).any():
        raise ValueError("Timestamps must be whole seconds.")
    finite_ts = finite_ts.astype("int64")
    ts_min, ts_max = int(finite_ts.min()), int(finite_ts.max())

    off_grid = (finite_ts - ts_min) % step_seconds != 0
    if off_grid.any():
        raise ValueError(
            f"{int(off_grid.sum())} rows have timestamps off the "
            f"{step_seconds}-second grid anchored at {ts_min}; refusing to "
            "resample silently."
        )

    valid = (
        ts_ok
        & np.isfinite(avg)
        & avg.between(CPU_MIN, CPU_MAX)
        & cpu["vm_id"].notna()
    )
    rows = cpu.loc[valid].copy()
    rows["timestamp"] = ts[valid].astype("int64")
    rows["avg_cpu"] = avg[valid].astype("float64")
    n_invalid = n_in - int(valid.sum())

    key = ["vm_id", "timestamp"]
    dup_mask = rows.duplicated(key, keep="first")
    n_dups = int(dup_mask.sum())
    n_conflicts = 0
    if n_dups:
        all_dups = rows[rows.duplicated(key, keep=False)]
        n_conflicts = int((all_dups.groupby(key)["avg_cpu"].nunique() > 1).sum())
        rows = rows.loc[~dup_mask]

    rows = rows.sort_values(["timestamp", "vm_id"], kind="mergesort").reset_index(
        drop=True
    )
    if rows.empty:
        raise ValueError("No valid CPU observations remain after cleaning.")

    return CleanedReadings(
        rows=rows,
        n_rows_in=n_in,
        n_invalid_rows_dropped=n_invalid,
        n_duplicate_rows_dropped=n_dups,
        n_conflicting_duplicate_keys=n_conflicts,
        raw_ts_min=ts_min,
        raw_ts_max=ts_max,
        step_seconds=step_seconds,
    )


def targets_from_cleaned(
    cleaned: CleanedReadings,
) -> tuple[pd.DataFrame, TargetBuildReport]:
    """Aggregate cleaned rows into the target series on the full 5-minute grid.

    Returns a frame indexed by ``timestamp`` (seconds, int64) with columns
    ``active_vm_count`` (int64), ``aggregate_cpu_utilization`` (float64, NaN at
    gaps) and ``is_gap`` (bool), plus a :class:`TargetBuildReport`.
    """
    step = cleaned.step_seconds
    grouped = cleaned.rows.groupby("timestamp", sort=True)["avg_cpu"]
    counts, means = grouped.size(), grouped.mean()

    grid = pd.Index(
        np.arange(cleaned.raw_ts_min, cleaned.raw_ts_max + step, step, dtype="int64"),
        name="timestamp",
    )
    series = pd.DataFrame(
        {
            TARGET_ACTIVE: counts.reindex(grid, fill_value=0).astype("int64"),
            TARGET_CPU: means.reindex(grid).astype("float64"),
        }
    )
    series["is_gap"] = series[TARGET_ACTIVE] == 0

    gaps = tuple(int(t) for t in series.index[series["is_gap"]])
    report = TargetBuildReport(
        n_rows_in=cleaned.n_rows_in,
        n_invalid_rows_dropped=cleaned.n_invalid_rows_dropped,
        n_duplicate_rows_dropped=cleaned.n_duplicate_rows_dropped,
        n_conflicting_duplicate_keys=cleaned.n_conflicting_duplicate_keys,
        n_rows_used=len(cleaned.rows),
        n_vms=int(cleaned.rows["vm_id"].nunique()),
        grid_start=cleaned.raw_ts_min,
        grid_end=cleaned.raw_ts_max,
        step_seconds=step,
        n_grid_points=len(series),
        n_gap_timestamps=len(gaps),
        gap_timestamps=gaps,
    )
    return series, report


def build_target_series(
    cpu: pd.DataFrame, step_seconds: int = STEP_SECONDS
) -> tuple[pd.DataFrame, TargetBuildReport]:
    """Convenience wrapper: clean raw readings, then build the target series."""
    return targets_from_cleaned(clean_readings(cpu, step_seconds))
