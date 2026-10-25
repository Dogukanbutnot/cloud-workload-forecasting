import numpy as np
import pandas as pd
import pytest

from cwfc.targets import (
    STEP_SECONDS,
    TARGET_ACTIVE,
    TARGET_CPU,
    build_target_series,
    clean_readings,
)


def make_cpu(n_vms=3, n_ts=4, start=0, value=None):
    """Tiny regular panel: VM i has avg_cpu = 10 * (i + 1) at every timestamp."""
    rows = []
    for t in range(n_ts):
        for i in range(n_vms):
            v = 10.0 * (i + 1) if value is None else value
            rows.append(
                {"timestamp": start + t * STEP_SECONDS, "vm_id": f"vm{i}",
                 "min_cpu": v / 2, "max_cpu": min(v * 2, 100.0), "avg_cpu": v}
            )
    return pd.DataFrame(rows)


def test_basic_counts_means_and_dtypes():
    series, report = build_target_series(make_cpu())
    assert list(series.index) == [0, 300, 600, 900]
    assert series.index.dtype == np.int64
    assert (series[TARGET_ACTIVE] == 3).all()
    assert series[TARGET_ACTIVE].dtype == np.int64
    assert np.allclose(series[TARGET_CPU], 20.0)  # mean of 10, 20, 30
    assert not series["is_gap"].any()
    assert report.n_rows_used == 12 and report.n_vms == 3
    assert report.n_gap_timestamps == 0


def test_duplicate_rows_counted_once_and_reported():
    cpu = make_cpu()
    dup = cpu.iloc[[0]].copy()
    dup["avg_cpu"] = 99.0  # conflicting duplicate of (vm0, t=0)
    series, report = build_target_series(pd.concat([cpu, dup], ignore_index=True))
    assert series.loc[0, TARGET_ACTIVE] == 3
    assert report.n_duplicate_rows_dropped == 1
    assert report.n_conflicting_duplicate_keys == 1
    # first occurrence wins: mean stays (10 + 20 + 30) / 3
    assert series.loc[0, TARGET_CPU] == pytest.approx(20.0)


def test_identical_duplicate_is_not_a_conflict():
    cpu = make_cpu()
    _, report = build_target_series(pd.concat([cpu, cpu.iloc[[0]]], ignore_index=True))
    assert report.n_duplicate_rows_dropped == 1
    assert report.n_conflicting_duplicate_keys == 0


def test_invalid_rows_dropped_before_dedup():
    cpu = make_cpu()
    bad_first = cpu.iloc[[0]].copy()
    bad_first["avg_cpu"] = np.nan
    # NaN row comes first, the valid row for the same key follows.
    frame = pd.concat([bad_first, cpu], ignore_index=True)
    series, report = build_target_series(frame)
    assert report.n_invalid_rows_dropped == 1
    assert report.n_duplicate_rows_dropped == 0
    assert series.loc[0, TARGET_ACTIVE] == 3  # the valid reading survived


@pytest.mark.parametrize("bad", [-1.0, 100.5, np.inf, -np.inf, np.nan])
def test_out_of_range_or_non_finite_avg_cpu_is_invalid(bad):
    cpu = make_cpu()
    cpu.loc[0, "avg_cpu"] = bad
    series, report = build_target_series(cpu)
    assert report.n_invalid_rows_dropped == 1
    assert series.loc[0, TARGET_ACTIVE] == 2
    assert series.loc[0, TARGET_CPU] == pytest.approx(25.0)  # mean of 20, 30


def test_boundary_values_0_and_100_are_valid():
    cpu = make_cpu(n_vms=2, n_ts=1)
    cpu["avg_cpu"] = [0.0, 100.0]
    series, report = build_target_series(cpu)
    assert report.n_invalid_rows_dropped == 0
    assert series.loc[0, TARGET_CPU] == pytest.approx(50.0)


def test_null_vm_id_is_invalid():
    cpu = make_cpu()
    cpu.loc[1, "vm_id"] = None
    series, report = build_target_series(cpu)
    assert report.n_invalid_rows_dropped == 1
    assert series.loc[0, TARGET_ACTIVE] == 2


def test_gap_is_flagged_not_interpolated():
    cpu = make_cpu(n_ts=5)
    cpu = cpu[cpu["timestamp"] != 600]  # whole timestamp missing from source
    series, report = build_target_series(cpu)
    assert list(series.index) == [0, 300, 600, 900, 1200]  # grid stays regular
    assert bool(series.loc[600, "is_gap"])
    assert series.loc[600, TARGET_ACTIVE] == 0
    assert np.isnan(series.loc[600, TARGET_CPU])  # never invented
    assert report.n_gap_timestamps == 1 and report.gap_timestamps == (600,)


def test_timestamp_with_only_invalid_rows_becomes_a_gap_inside_the_grid():
    cpu = make_cpu(n_ts=3)
    cpu.loc[cpu["timestamp"] == 300, "avg_cpu"] = np.nan
    series, _ = build_target_series(cpu)
    assert bool(series.loc[300, "is_gap"]) and np.isnan(series.loc[300, TARGET_CPU])


def test_off_grid_timestamps_raise():
    cpu = make_cpu()
    cpu.loc[0, "timestamp"] = 17
    with pytest.raises(ValueError, match="off the 300-second grid"):
        build_target_series(cpu)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        build_target_series(make_cpu().drop(columns=["avg_cpu"]))


def test_all_invalid_raises():
    cpu = make_cpu()
    cpu["avg_cpu"] = np.nan
    with pytest.raises(ValueError, match="No valid CPU observations"):
        build_target_series(cpu)


def test_non_zero_start_and_input_order_do_not_matter():
    cpu = make_cpu(start=10_000)
    shuffled = cpu.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a, _ = build_target_series(cpu)
    b, _ = build_target_series(shuffled)
    pd.testing.assert_frame_equal(a, b)
    assert a.index[0] == 10_000


def test_clean_readings_does_not_mutate_input():
    cpu = make_cpu()
    before = cpu.copy()
    clean_readings(cpu)
    pd.testing.assert_frame_equal(cpu, before)
