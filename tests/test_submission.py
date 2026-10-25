import pandas as pd
import pytest

from cwfc.submission import COLUMNS, contract_problems, validate_submission, write_submission

TS = [1000 + 300 * i for i in range(24)]


def good():
    return pd.DataFrame({"timestamp": TS, "active_vm_count": 10000.0, "aggregate_cpu_utilization": 9.4})


def test_valid_frame_has_no_problems():
    assert contract_problems(good(), TS) == []
    validate_submission(good(), TS)


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda d: d.iloc[:23], "expected 24 rows"),
        (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "expected 24 rows"),
        (lambda d: d.assign(timestamp=[TS[0]] + TS[:-1]), "duplicate timestamps"),
        (lambda d: d.iloc[::-1].reset_index(drop=True), "do not match the holdout timestamps"),
        (lambda d: d.assign(timestamp=[t + 1 for t in TS]), "do not match the holdout timestamps"),
        (lambda d: d.assign(aggregate_cpu_utilization=float("nan")), "missing or non-finite"),
        (lambda d: d.assign(active_vm_count=float("inf")), "missing or non-finite"),
        (lambda d: d.assign(active_vm_count=-1.0), "active_vm_count must be >= 0"),
        (lambda d: d.assign(aggregate_cpu_utilization=100.5), "must be in [0, 100]"),
        (lambda d: d.assign(aggregate_cpu_utilization=-0.1), "must be in [0, 100]"),
        (lambda d: d[["active_vm_count", "timestamp", "aggregate_cpu_utilization"]], "columns must be exactly"),
        (lambda d: d.assign(extra=1), "columns must be exactly"),
        (lambda d: d.rename(columns={"timestamp": "ts"}), "columns must be exactly"),
    ],
)
def test_each_contract_violation_is_reported(mutate, fragment):
    problems = contract_problems(mutate(good()), TS)
    assert any(fragment in p for p in problems), problems
    with pytest.raises(ValueError, match="invalid submission"):
        validate_submission(mutate(good()), TS)


def test_boundary_values_are_accepted():
    df = good().assign(active_vm_count=0.0, aggregate_cpu_utilization=100.0)
    assert contract_problems(df, TS) == []
    assert contract_problems(df.assign(aggregate_cpu_utilization=0.0), TS) == []


def test_write_submission_validates_and_writes_no_index(tmp_path):
    path = write_submission(good(), tmp_path / "sub" / "predictions.csv", TS)
    back = pd.read_csv(path)
    assert list(back.columns) == COLUMNS and len(back) == 24
    assert path.read_text().splitlines()[0] == ",".join(COLUMNS)  # no index column
    with pytest.raises(ValueError):
        write_submission(good().iloc[:5], tmp_path / "bad.csv", TS)
    assert not (tmp_path / "bad.csv").exists()
