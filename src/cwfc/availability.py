"""Which columns may be used as model inputs at a forecast origin?

ASSIGNMENT.md: "A feature is valid only if it could have been computed at the
forecast origin." The exploratory analysis found that three metadata columns
violate this:

* ``avg_cpu``, ``max_cpu`` and ``p95_max_cpu`` in ``vm_metadata`` are *VM-level
  statistics over the VM's whole life* (the source trace spans 30 days, the
  challenge window only ~11 hours). They therefore contain information from
  after any forecast origin, including the holdout horizon.

They are not wrong data, and descriptive analysis may use them (see
``cwfc.eda.metadata_representativeness``), but a model must not. The same
statistics recomputed from readings *up to the origin* are valid features.

Static attributes known when a VM is created (category, core and memory
buckets, subscription and deployment ids) are available at any origin.
"""

from __future__ import annotations

from typing import Iterable

LIFETIME_METADATA_COLUMNS = frozenset({"avg_cpu", "max_cpu", "p95_max_cpu"})
STATIC_METADATA_COLUMNS = frozenset(
    {"vm_category", "virtual_core_bucket", "memory_gb_bucket", "subscription_id", "deployment_id"}
)
# Deletion time is a *future* event relative to every origin before it happens.
FUTURE_METADATA_COLUMNS = frozenset({"timestamp_vm_deleted"})


def forbidden_feature_columns(columns: Iterable[str]) -> list[str]:
    """Columns in ``columns`` that would leak information from after the origin."""
    banned = LIFETIME_METADATA_COLUMNS | FUTURE_METADATA_COLUMNS
    return sorted(c for c in columns if c in banned)


def assert_features_available_at_origin(columns: Iterable[str]) -> None:
    """Raise if any column is a lifetime statistic or a future event."""
    bad = forbidden_feature_columns(columns)
    if bad:
        raise ValueError(
            f"features {bad} are computed over the VM's whole lifetime (or are "
            "future events) and are not available at the forecast origin"
        )
