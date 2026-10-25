import pandas as pd
import pytest

from cwfc.population import check_constant_population, fixed_population_forecast
from cwfc.targets import STEP_SECONDS, TARGET_ACTIVE, clean_readings, targets_from_cleaned

N_VMS, N_TS, N_TRAIN = 5, 10, 7
LAST_TS = (N_TS - 1) * STEP_SECONDS


def panel(drop=None):
    rows = [
        {"timestamp": t * STEP_SECONDS, "vm_id": f"vm{i}", "min_cpu": 1.0,
         "max_cpu": 9.0, "avg_cpu": 5.0}
        for t in range(N_TS) for i in range(N_VMS)
    ]
    df = pd.DataFrame(rows)
    return df if drop is None else df[~drop(df)]


def metadata(created=0, deleted=10**9):
    return pd.DataFrame(
        {
            "vm_id": [f"vm{i}" for i in range(N_VMS)],
            "timestamp_vm_created": created,
            "timestamp_vm_deleted": deleted,
        }
    )


def run(cpu, meta, horizon_end=LAST_TS):
    cleaned = clean_readings(cpu)
    series, _ = targets_from_cleaned(cleaned)
    train = series.iloc[:N_TRAIN]
    return check_constant_population(train, cleaned.rows, meta, horizon_end), train


def test_fixed_population_is_proven_and_forecast_is_constant():
    report, train = run(panel(), metadata())
    assert report.is_constant and report.constant_value == N_VMS
    assert report.n_vms_full_coverage == N_VMS and report.n_vms_partial_coverage == 0
    assert report.vm_sets_match
    assert report.fixed_by_construction
    future = pd.Index([(N_TRAIN + k) * STEP_SECONDS for k in range(3)])
    fc = fixed_population_forecast(report, future)
    assert (fc == N_VMS).all() and list(fc.index) == list(future)


def test_vm_missing_from_training_window_means_not_constant():
    # vm0 has no readings at the last two training timestamps (a VM that died)
    cpu = panel(drop=lambda d: (d.vm_id == "vm0") & (d.timestamp >= 5 * STEP_SECONDS))
    report, _ = run(cpu, metadata())
    assert not report.is_constant and report.constant_value is None
    assert report.n_vms_partial_coverage == 1
    assert report.active_min == N_VMS - 1 and report.active_max == N_VMS
    assert not report.fixed_by_construction
    assert "genuine forecasting target" in report.verdict


def test_vm_deleted_inside_the_horizon_is_caught_by_metadata():
    # Constant on the training window, but vm0 is deleted before the horizon
    # ends. A constant forecast would be wrong; metadata must veto it.
    meta = metadata()
    meta.loc[0, "timestamp_vm_deleted"] = LAST_TS - STEP_SECONDS
    report, _ = run(panel(), meta)
    assert report.is_constant
    assert report.n_deleted_before_horizon_end == 1
    assert not report.fixed_by_construction
    with pytest.raises(ValueError, match="not proven fixed"):
        fixed_population_forecast(report, pd.Index([0]))


def test_deletion_exactly_at_horizon_end_is_allowed():
    meta = metadata()
    meta.loc[0, "timestamp_vm_deleted"] = LAST_TS
    report, _ = run(panel(), meta)
    assert report.n_deleted_before_horizon_end == 0 and report.fixed_by_construction


def test_vm_created_after_window_start_is_not_fixed():
    meta = metadata()
    meta.loc[0, "timestamp_vm_created"] = STEP_SECONDS
    report, _ = run(panel(), meta)
    assert report.n_created_after_window_start == 1
    assert not report.fixed_by_construction


def test_unknown_creation_time_is_treated_as_failure_but_missing_deletion_is_alive():
    meta = metadata()
    meta.loc[0, "timestamp_vm_created"] = float("nan")
    report, _ = run(panel(), meta)
    assert report.n_created_after_window_start == 1 and not report.fixed_by_construction

    meta = metadata()
    meta["timestamp_vm_deleted"] = float("nan")  # never deleted
    report, _ = run(panel(), meta)
    assert report.n_deleted_before_horizon_end == 0 and report.fixed_by_construction


def test_without_metadata_constant_window_is_not_enough():
    report, _ = run(panel(), None)
    assert report.is_constant and not report.metadata_checked
    assert not report.fixed_by_construction
    assert "no metadata" in report.verdict


def test_metadata_and_cpu_vm_sets_must_match():
    meta = metadata().iloc[:-1]  # metadata lacks vm4
    report, _ = run(panel(), meta)
    assert report.vm_sets_match is False and not report.fixed_by_construction


def test_missing_metadata_columns_raise():
    cleaned = clean_readings(panel())
    series, _ = targets_from_cleaned(cleaned)
    with pytest.raises(ValueError, match="missing required columns"):
        check_constant_population(series, cleaned.rows, metadata().drop(columns=["vm_id"]))


def test_report_uses_only_the_window_it_is_given():
    # Churn that happens only AFTER the training window must not change the
    # window-level result: the check never inspects holdout target values.
    cpu = panel(drop=lambda d: (d.vm_id == "vm0") & (d.timestamp >= N_TRAIN * STEP_SECONDS))
    report, train = run(cpu, metadata())
    assert (train[TARGET_ACTIVE] == N_VMS).all()
    assert report.is_constant and report.fixed_by_construction


def test_report_serialises_to_dict():
    report, _ = run(panel(), metadata())
    d = report.to_dict()
    assert d["fixed_by_construction"] is True and d["constant_value"] == N_VMS
