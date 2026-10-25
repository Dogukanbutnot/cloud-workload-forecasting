import pandas as pd
import pytest

from cwfc.rolling import rolling_origin_folds
from cwfc.validation import rolling_origin_splits


def test_official_configuration_gives_seven_folds_ending_at_the_window_end():
    folds = rolling_origin_folds(110, horizon=24, min_train_size=50, step=6)
    assert [f.train_size for f in folds] == [50, 56, 62, 68, 74, 80, 86]
    assert folds[-1].val_stop == 110  # never reaches into the holdout (positions >= 110)
    assert [f.fold for f in folds] == list(range(7))


def test_expanding_window_and_no_train_validation_overlap():
    for f in rolling_origin_folds(110, 24, 50, 6):
        assert f.val_start == f.train_size and f.horizon == 24
        assert f.val_stop <= 110


def test_step_defaults_to_horizon_and_windows_do_not_overlap():
    folds = rolling_origin_folds(60, horizon=10, min_train_size=20)
    assert [f.train_size for f in folds] == [20, 30, 40, 50]
    for a, b in zip(folds, folds[1:]):
        assert a.val_stop <= b.val_start


def test_exact_fit_yields_a_single_fold():
    folds = rolling_origin_folds(30, horizon=10, min_train_size=20)
    assert len(folds) == 1 and folds[0].val_stop == 30


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_obs=29, horizon=10, min_train_size=20),  # too short
        dict(n_obs=110, horizon=288, min_train_size=2016),  # starter defaults
        dict(n_obs=110, horizon=0, min_train_size=10),
        dict(n_obs=110, horizon=10, min_train_size=0),
        dict(n_obs=110, horizon=10, min_train_size=10, step=0),
    ],
)
def test_unusable_configuration_raises_instead_of_returning_nothing(kwargs):
    with pytest.raises(ValueError):
        rolling_origin_folds(**kwargs)


def test_matches_starter_splits_once_timestamps_are_converted_correctly():
    n = 110
    ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    starter = rolling_origin_splits(
        pd.DataFrame({"timestamp": ts}), horizon=24, min_train_size=50, step=6
    )
    mine = rolling_origin_folds(n, horizon=24, min_train_size=50, step=6)
    assert len(starter) == len(mine)
    for s, f in zip(starter, mine):
        assert s.train_end == ts[f.train_size - 1]
        assert s.validation_start == ts[f.val_start]
        assert s.validation_end == ts[f.val_stop - 1]
