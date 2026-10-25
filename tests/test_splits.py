import numpy as np
import pandas as pd
import pytest

from cwfc.splits import (
    HORIZON,
    OFFICIAL_SERIES_LENGTH,
    OFFICIAL_TRAIN_SIZE,
    is_official_protocol,
    split_train_holdout,
)
from cwfc.targets import TARGET_ACTIVE, TARGET_CPU


def make_series(n, step=300):
    idx = pd.Index(np.arange(n, dtype="int64") * step, name="timestamp")
    return pd.DataFrame(
        {TARGET_ACTIVE: 100, TARGET_CPU: np.arange(n, dtype=float), "is_gap": False},
        index=idx,
    )


def test_official_shape_is_110_24():
    split = split_train_holdout(make_series(OFFICIAL_SERIES_LENGTH))
    assert (split.train_size, split.horizon) == (OFFICIAL_TRAIN_SIZE, HORIZON)
    assert is_official_protocol(split)


def test_split_is_chronological_contiguous_and_complete():
    s = make_series(134)
    split = split_train_holdout(s)
    assert split.train.index.max() < split.holdout_timestamps.min()
    assert split.holdout_timestamps.equals(s.index[-24:])  # final 24 timestamps
    # holdout is contiguous at the 5-minute cadence, in order, no duplicates
    assert (np.diff(split.holdout_timestamps.to_numpy()) == 300).all()
    # train + holdout reconstruct the full series, nothing lost or reordered
    full = pd.concat([split.train, split.reveal_holdout()])
    pd.testing.assert_frame_equal(full, s)


def test_not_hard_coded_to_134_for_smoke_datasets():
    split = split_train_holdout(make_series(40))
    assert (split.train_size, split.horizon) == (16, 24)
    assert not is_official_protocol(split)


def test_custom_horizon():
    split = split_train_holdout(make_series(50), horizon=6)
    assert (split.train_size, split.horizon) == (44, 6)


def test_too_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        split_train_holdout(make_series(24))  # nothing left to train on
    with pytest.raises(ValueError, match="too short"):
        split_train_holdout(make_series(30), min_train_size=10)


def test_invalid_horizon_raises():
    with pytest.raises(ValueError, match="positive"):
        split_train_holdout(make_series(30), horizon=0)


def test_unsorted_or_duplicate_index_raises():
    s = make_series(40)
    with pytest.raises(ValueError, match="chronological"):
        split_train_holdout(s.iloc[::-1])
    dup = pd.concat([s, s.iloc[[-1]]])
    with pytest.raises(ValueError, match="chronological"):
        split_train_holdout(dup)


def test_holdout_values_are_not_in_train_and_not_in_repr():
    s = make_series(134)
    s.loc[s.index[-24:], TARGET_CPU] = 777.0  # marker only present in holdout
    split = split_train_holdout(s)
    assert not (split.train[TARGET_CPU] == 777.0).any()
    assert "777" not in repr(split)
    assert (split.reveal_holdout()[TARGET_CPU] == 777.0).all()


def test_reveal_holdout_returns_a_copy_and_train_is_decoupled():
    s = make_series(134)
    split = split_train_holdout(s)
    tampered = split.reveal_holdout()
    tampered[TARGET_CPU] = -1.0
    assert (split.reveal_holdout()[TARGET_CPU] != -1.0).all()
    split.train.loc[split.train.index[0], TARGET_CPU] = -5.0
    assert s.iloc[0][TARGET_CPU] == 0.0  # caller's frame untouched
