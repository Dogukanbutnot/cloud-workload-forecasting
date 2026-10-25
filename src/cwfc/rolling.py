"""Rolling-origin (expanding-window) fold generation, by position.

Why this exists next to ``cwfc.validation.rolling_origin_splits``:

* The starter defaults (``horizon=288``, ``min_train_size=2016``) yield **zero**
  folds on the 110-point training window, and it returns an empty list
  silently. Here an unusable configuration raises.
* The starter converts timestamps with ``pd.to_datetime(..., utc=True)``, which
  reads integer *seconds* as *nanoseconds*, so its fold boundaries come out as
  1970-01-01 microseconds. Ordering survives, meaning is lost. Working with
  integer positions avoids any unit ambiguity; timestamps stay in the data.

``tests/test_rolling.py`` checks that, once timestamps are converted correctly,
these folds coincide with the starter's.

Design on the official data (110 training points, horizon 24, minimum train 50,
step 6): origins at train sizes 50, 56, ..., 86, i.e. 7 folds. The last
validation window ends exactly at the end of the training window, so no fold can
ever reach into the holdout. Neighbouring folds overlap in their validation
windows, so fold errors are positively correlated: they are repeated looks at
the same stretch of series, not independent replications. Reports should say so
and show the spread across folds instead of a single average.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingFold:
    """One forecast origin: history is ``[0, train_size)``, scored on the next
    ``horizon`` observations ``[val_start, val_stop)``."""

    fold: int
    train_size: int
    val_start: int
    val_stop: int

    @property
    def horizon(self) -> int:
        return self.val_stop - self.val_start


def rolling_origin_folds(
    n_obs: int,
    horizon: int,
    min_train_size: int,
    step: int | None = None,
) -> list[RollingFold]:
    """Expanding-window folds over ``n_obs`` observations.

    ``step`` defaults to ``horizon`` (non-overlapping validation windows). A
    configuration that produces no fold raises ``ValueError``.
    """
    if min(horizon, min_train_size) <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    step = horizon if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    if n_obs < min_train_size + horizon:
        raise ValueError(
            f"{n_obs} observations cannot supply min_train_size={min_train_size} "
            f"plus horizon={horizon}: no rolling-origin fold is possible"
        )

    folds: list[RollingFold] = []
    train_size = min_train_size
    while train_size + horizon <= n_obs:
        folds.append(
            RollingFold(
                fold=len(folds),
                train_size=train_size,
                val_start=train_size,
                val_stop=train_size + horizon,
            )
        )
        train_size += step
    return folds
