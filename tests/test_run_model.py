import pytest

from cwfc import run_model
from smoke import N_TS, write_smoke_dataset

HORIZON = 6
TRAIN = N_TS - HORIZON


def test_model_evaluation_never_reveals_the_holdout_and_stays_inside_the_training_window(tmp_path, never_reveal):
    write_smoke_dataset(tmp_path)
    res = run_model.run(tmp_path, horizon=HORIZON, min_train_size=30, step=6, output_dir=tmp_path / "out")
    assert res["predictions"]["timestamp"].max() < TRAIN * 300  # strictly inside the training window
    assert res["folds"][-1].val_stop == TRAIN
    assert {"mean_reverting_level", "damped_persistence", "persistence", "history_mean"} <= set(res["summary"]["model"])
    assert len(res["params"]) == len(res["folds"])
    assert (res["params"]["phi"] < 1).all()
    for name in ("model_summary.csv", "model_error_by_band.csv", "model_error_by_horizon.csv",
                 "model_fold_mae.csv", "model_parameters_by_fold.csv", "model_regret_vs_best_baseline.csv"):
        assert (tmp_path / "out" / name).exists()


def test_envelope_and_regret_are_consistent_with_the_fold_table(tmp_path):
    write_smoke_dataset(tmp_path)
    res = run_model.run(tmp_path, horizon=HORIZON, min_train_size=30, step=6)
    assert "best_baseline_per_step" in res["by_horizon"].columns
    reg = res["regret"].set_index("model")
    assert reg.loc["persistence", "max_regret"] >= 0  # a baseline is never better than the best baseline
    assert (res["fold_mae"].index == [f.fold for f in res["folds"]]).all()


def test_paired_comparison_reports_fold_counts(tmp_path):
    write_smoke_dataset(tmp_path)
    res = run_model.run(tmp_path, horizon=HORIZON, min_train_size=30, step=6)
    p = run_model.paired_vs(res["fold_mae"], "damped_persistence", "persistence")
    assert p["n_folds"] == len(res["folds"]) and 0 <= p["folds_model_better"] <= p["n_folds"]


def test_unusable_fold_configuration_fails_loudly(tmp_path):
    write_smoke_dataset(tmp_path)
    with pytest.raises(ValueError, match="no rolling-origin fold"):
        run_model.run(tmp_path, horizon=HORIZON, min_train_size=2016, step=6)


def test_comparison_figure_is_written_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    write_smoke_dataset(tmp_path)
    run_model.run(tmp_path, horizon=HORIZON, min_train_size=30, step=6, output_dir=tmp_path / "out")
    assert (tmp_path / "out" / "fig_model_comparison.png").stat().st_size > 1000
