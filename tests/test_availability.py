import pytest

from cwfc.availability import (
    LIFETIME_METADATA_COLUMNS,
    STATIC_METADATA_COLUMNS,
    assert_features_available_at_origin,
    forbidden_feature_columns,
)


def test_lifetime_statistics_and_future_events_are_forbidden():
    cols = ["vm_category", "avg_cpu", "p95_max_cpu", "max_cpu", "timestamp_vm_deleted", "lag_1"]
    assert forbidden_feature_columns(cols) == ["avg_cpu", "max_cpu", "p95_max_cpu", "timestamp_vm_deleted"]
    with pytest.raises(ValueError, match="not available at the forecast origin"):
        assert_features_available_at_origin(cols)


def test_static_attributes_and_history_derived_features_pass():
    assert_features_available_at_origin(
        list(STATIC_METADATA_COLUMNS) + ["lag_1", "rolling_mean_12", "hist_avg_cpu"]
    )


def test_the_two_sets_do_not_overlap():
    assert not (LIFETIME_METADATA_COLUMNS & STATIC_METADATA_COLUMNS)
