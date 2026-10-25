import pandas as pd
import pytest

from cwfc import build_targets, run_baselines
from smoke import N_TS, write_smoke_dataset

HORIZON = 6  # a tiny "smoke" dataset, nothing like 134 / 24


def test_build_targets_never_reveals_the_holdout_and_does_not_hard_code_134(tmp_path, never_reveal):
    write_smoke_dataset(tmp_path)
    s = build_targets.run(tmp_path, horizon=HORIZON, output_dir=tmp_path / "out")
    assert s["split"]["train_size"] == N_TS - HORIZON and s["split"]["horizon"] == HORIZON
    assert s["split"]["official_110_24"] is False
    assert s["population"]["fixed_by_construction"] is True
    # holdout targets are not written anywhere
    train_csv = pd.read_csv(tmp_path / "out" / "targets_train.csv")
    assert len(train_csv) == N_TS - HORIZON
    assert list(pd.read_csv(tmp_path / "out" / "holdout_timestamps.csv").iloc[:, 0]) == [
        t * 300 for t in range(N_TS - HORIZON, N_TS)
    ]


def test_baseline_pipeline_never_reveals_the_holdout(tmp_path, never_reveal):
    write_smoke_dataset(tmp_path)
    res = run_baselines.run(tmp_path, horizon=HORIZON, min_train_size=30, step=6, output_dir=tmp_path / "out")
    folds = res["folds"]
    assert [f.train_size for f in folds] == [30, 36, 42, 48]
    assert folds[-1].val_stop == N_TS - HORIZON  # validation ends where training ends
    for target, parts in res["targets"].items():
        assert parts["predictions"]["timestamp"].max() < (N_TS - HORIZON) * 300  # strictly inside train
        assert not parts["summary"].empty
    assert (tmp_path / "out" / "baseline_summary_aggregate_cpu_utilization.csv").exists()


def test_active_vm_count_is_exact_and_fixed_population_is_validated_per_fold(tmp_path):
    write_smoke_dataset(tmp_path)
    res = run_baselines.run(tmp_path, horizon=HORIZON, min_train_size=30, step=6)
    s = res["targets"]["active_vm_count"]["summary"].set_index("model")
    assert "fixed_population" in s.index
    assert (s["mae"] == 0).all()


def test_unproven_population_drops_the_constant_forecaster_but_still_runs(tmp_path):
    # vm0 is constant on the training window but deleted inside the holdout
    write_smoke_dataset(tmp_path, delete_vm0_at=(N_TS - 2) * 300)
    res = run_baselines.run(tmp_path, horizon=HORIZON, min_train_size=30, step=6)
    assert res["population"].fixed_by_construction is False
    s = res["targets"]["active_vm_count"]["summary"]
    assert "fixed_population" not in set(s["model"])
    assert "persistence" in set(s["model"])


def test_misconfigured_folds_fail_loudly(tmp_path):
    write_smoke_dataset(tmp_path)
    with pytest.raises(ValueError, match="no rolling-origin fold"):
        run_baselines.run(tmp_path, horizon=HORIZON, min_train_size=2016, step=6)
