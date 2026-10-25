"""Chronological train / holdout split of the aggregate target series.

Protocol (ASSIGNMENT.md, section 3C): the first 110 observations are the public
training window and the final 24 observations (2 hours at 5-minute cadence) are
the held-out forecast horizon.

Design choices:

* The split is defined from the *end*: the last ``horizon`` observations are the
  holdout and everything before them is training. On the official 134-point
  series this is exactly 110 / 24, but the code does not hard-code 134, so the
  pipeline also runs on the small smoke dataset reviewers may substitute
  (SUBMISSION.md, "What reviewers will run").
* The split is positional and never shuffled. Ordering and uniqueness of the
  index are verified, not assumed.
* Holdout *values* are hidden behind :meth:`TrainHoldoutSplit.reveal_holdout`.
  Everything needed to produce ``predictions.csv`` (the holdout timestamps) is
  available without touching them. Modelling, feature selection, tuning and
  model selection code must only ever see ``split.train``; the single legitimate
  call to ``reveal_holdout`` is the final, one-shot evaluation. Making that call
  explicit keeps accidental peeking out of exploratory code and makes the
  protocol auditable by grepping for the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

HORIZON = 24
OFFICIAL_TRAIN_SIZE = 110
OFFICIAL_SERIES_LENGTH = OFFICIAL_TRAIN_SIZE + HORIZON  # 134


@dataclass(frozen=True)
class TrainHoldoutSplit:
    """Training window plus the holdout *timestamps*; holdout values are guarded."""

    train: pd.DataFrame
    holdout_timestamps: pd.Index
    _holdout: pd.DataFrame = field(repr=False)

    @property
    def horizon(self) -> int:
        return len(self.holdout_timestamps)

    @property
    def train_size(self) -> int:
        return len(self.train)

    def reveal_holdout(self) -> pd.DataFrame:
        """Return holdout target values. Call only for the final evaluation."""
        return self._holdout.copy()


def split_train_holdout(
    series: pd.DataFrame,
    horizon: int = HORIZON,
    min_train_size: int = 1,
) -> TrainHoldoutSplit:
    """Split the *full* target series into a training window and a holdout.

    ``series`` must already be built on the complete timeline (see
    :func:`cwfc.targets.build_target_series`); the split is applied afterwards
    so no training-time quantity is ever derived using holdout information.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    idx = series.index
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("series index must be strictly increasing (chronological)")
    n = len(series)
    if n - horizon < min_train_size:
        raise ValueError(
            f"series of length {n} is too short for horizon={horizon} and "
            f"min_train_size={min_train_size}"
        )

    cut = n - horizon
    return TrainHoldoutSplit(
        train=series.iloc[:cut].copy(),
        holdout_timestamps=series.index[cut:].copy(),
        _holdout=series.iloc[cut:].copy(),
    )


def is_official_protocol(split: TrainHoldoutSplit) -> bool:
    """True iff the split has the official 110 / 24 shape."""
    return (
        split.train_size == OFFICIAL_TRAIN_SIZE and split.horizon == HORIZON
    )
