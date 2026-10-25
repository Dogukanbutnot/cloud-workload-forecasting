"""Is ``active_vm_count`` a forecasting problem at all, or a constant?

``active_vm_count(t)`` counts VMs with a valid reading at ``t``. The data card
states that only VMs with a *complete* 134-point trace were eligible for the
release. That is a selection on the outcome (survivorship): by construction no
VM can appear or disappear inside the observation window, so the count is
expected to be flat. This module turns that expectation into checks that can be
run, reported, and unit-tested.

Leakage discipline. The checks are meant to be run on the **training window**
plus the VM **metadata**, never on holdout targets:

1. ``active_vm_count`` is constant on the training window;
2. every VM has full coverage of the training window (no partial traces);
3. the metadata VM set equals the CPU VM set;
4. every VM was created at or before the window start (left-censored), and
5. every VM is still alive at ``horizon_end`` (its deletion timestamp is not
   before the last timestamp we must forecast).

(4) and (5) use only metadata and the timestamp grid, i.e. nothing derived from
holdout target values. Together they justify forecasting the constant over the
holdout *a priori*: no VM can be born or die inside the horizon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .targets import TARGET_ACTIVE

META_REQUIRED = ("vm_id", "timestamp_vm_created", "timestamp_vm_deleted")


@dataclass(frozen=True)
class PopulationReport:
    window_start: int
    window_end: int
    horizon_end: int
    n_timestamps: int
    n_vms_in_window: int
    active_min: int
    active_max: int
    is_constant: bool
    constant_value: int | None
    n_vms_full_coverage: int
    n_vms_partial_coverage: int
    metadata_checked: bool
    vm_sets_match: bool | None
    n_created_after_window_start: int | None
    n_deleted_before_horizon_end: int | None
    fixed_by_construction: bool
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def check_constant_population(
    series_window: pd.DataFrame,
    cleaned_rows: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    horizon_end: int | None = None,
) -> PopulationReport:
    """Test whether the VM population is fixed over the window and the horizon.

    Parameters
    ----------
    series_window:
        Target frame restricted to the window to inspect (the *training* window
        in the real pipeline), as produced by :mod:`cwfc.targets`.
    cleaned_rows:
        Cleaned VM-level rows (``CleanedReadings.rows``). Rows outside the
        window are ignored.
    meta:
        VM metadata with ``vm_id``, ``timestamp_vm_created`` and
        ``timestamp_vm_deleted``. Without it the population can be shown
        constant on the window but never *fixed by construction*.
    horizon_end:
        Last timestamp that must be forecast (end of the holdout). Defaults to
        the window end, in which case only the window itself is covered.
    """
    if series_window.empty:
        raise ValueError("series_window must not be empty")

    start = int(series_window.index.min())
    end = int(series_window.index.max())
    h_end = end if horizon_end is None else int(horizon_end)
    n_ts = len(series_window)

    active = series_window[TARGET_ACTIVE]
    is_constant = bool(active.nunique() == 1)
    constant_value = int(active.iloc[0]) if is_constant else None

    in_window = cleaned_rows[
        (cleaned_rows["timestamp"] >= start) & (cleaned_rows["timestamp"] <= end)
    ]
    per_vm = in_window.groupby("vm_id").size()
    n_full = int((per_vm == n_ts).sum())
    n_partial = int(len(per_vm) - n_full)

    metadata_checked = meta is not None
    vm_sets_match = n_created_after = n_deleted_before = None
    if meta is not None:
        missing = [c for c in META_REQUIRED if c not in meta.columns]
        if missing:
            raise ValueError(f"metadata is missing required columns: {missing}")
        vm_sets_match = set(meta["vm_id"]) == set(per_vm.index)
        created = pd.to_numeric(meta["timestamp_vm_created"], errors="coerce")
        deleted = pd.to_numeric(meta["timestamp_vm_deleted"], errors="coerce")
        # Unknown creation time is treated as a failure (conservative); an
        # absent deletion time means "not deleted", i.e. still alive.
        n_created_after = int((~(created <= start)).sum())
        n_deleted_before = int((deleted < h_end).sum())

    fixed = bool(
        is_constant
        and n_partial == 0
        and metadata_checked
        and vm_sets_match
        and n_created_after == 0
        and n_deleted_before == 0
    )

    if fixed:
        verdict = (
            f"Population is fixed by construction at {constant_value} VMs: "
            "constant on the window, full coverage for every VM, and no VM is "
            "created after the window start or deleted before the horizon end."
        )
    elif is_constant and not metadata_checked:
        verdict = (
            f"active_vm_count is constant ({constant_value}) on the window, but "
            "no metadata was supplied, so the horizon is not covered."
        )
    elif is_constant:
        verdict = (
            f"active_vm_count is constant ({constant_value}) on the window, but "
            "metadata or coverage checks failed: the population may change "
            "inside the horizon. Model it instead of assuming a constant."
        )
    else:
        verdict = (
            f"active_vm_count varies on the window ({int(active.min())} to "
            f"{int(active.max())}); it is a genuine forecasting target."
        )

    return PopulationReport(
        window_start=start,
        window_end=end,
        horizon_end=h_end,
        n_timestamps=n_ts,
        n_vms_in_window=int(len(per_vm)),
        active_min=int(active.min()),
        active_max=int(active.max()),
        is_constant=is_constant,
        constant_value=constant_value,
        n_vms_full_coverage=n_full,
        n_vms_partial_coverage=n_partial,
        metadata_checked=metadata_checked,
        vm_sets_match=vm_sets_match,
        n_created_after_window_start=n_created_after,
        n_deleted_before_horizon_end=n_deleted_before,
        fixed_by_construction=fixed,
        verdict=verdict,
    )


def fixed_population_forecast(
    report: PopulationReport, index: pd.Index
) -> pd.Series:
    """Forecast ``active_vm_count`` as the proven constant, or refuse.

    Refusing when the population is *not* proven fixed prevents a silent
    constant forecast on a dataset (e.g. a reviewer's smoke set) where the
    assumption does not hold.
    """
    if not report.fixed_by_construction or report.constant_value is None:
        raise ValueError(
            "Population is not proven fixed; fit a model for active_vm_count "
            "instead of forecasting a constant."
        )
    return pd.Series(float(report.constant_value), index=index, dtype=float)
