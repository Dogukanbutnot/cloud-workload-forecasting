import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import mean_forecast, persistence_forecast
from cwfc.evaluation import error_by_horizon, run_rolling_origin, summarize
from cwfc.rolling import RollingFold, rolling_origin_folds


def series(values, step=300):
    idx = pd.Index(np.arange(len(values), dtype="int64") * step, name="timestamp")
    return pd.DataFrame({"x": np.asarray(values, dtype=float)}, index=idx)


SUITE = {"persistence": persistence_forecast, "history_mean": mean_forecast}


def hand_example():
    # 8 points 0..7, one fold: history 0..5, validate on [6, 7]
    train = series(range(8))
    folds = rolling_origin_folds(8, horizon=2, min_train_size=6)
    assert len(folds) == 1
    return train, folds


def test_metrics_match_hand_computation():
    train, folds = hand_example()
    s = summarize(run_rolling_origin(train, "x", SUITE, folds)).set_index("model")

    p = s.loc["persistence"]  # forecast [5, 5] vs actual [6, 7]
    assert p["mae"] == pytest.approx(1.5)
    assert p["rmse"] == pytest.approx(np.sqrt(2.5))
    assert p["bias"] == pytest.approx(-1.5)  # forecast minus actual
    assert p["mase"] == pytest.approx(1.5)  # history one-step naive MAE is 1
    assert p["peak_mae"] == pytest.approx(1.5) and p["n_peak_points"] == 2  # 0.9-quantile of 0..5 is 4.5
    assert p["mae_ratio_vs_persistence"] == pytest.approx(1.0)

    m = s.loc["history_mean"]  # forecast [2.5, 2.5]: errors 3.5 and 4.5
    assert m["mae"] == pytest.approx(4.0)
    assert m["mae_ratio_vs_persistence"] == pytest.approx(4.0 / 1.5)
    assert m["folds_beating_persistence"] == 0 and m["n_folds"] == 1
    assert list(s.index) == ["persistence", "history_mean"]  # sorted by MAE


def test_prediction_table_shape_and_columns():
    train = series(np.arange(110, dtype=float))
    folds = rolling_origin_folds(110, 24, 50, 6)
    preds = run_rolling_origin(train, "x", SUITE, folds)
    assert len(preds) == len(folds) * len(SUITE) * 24
    assert {"fold", "train_size", "model", "step", "timestamp", "y_true", "y_pred",
            "mase_scale", "peak_threshold"} <= set(preds.columns)
    assert preds["step"].min() == 1 and preds["step"].max() == 24
    assert preds["timestamp"].max() == train.index[-1]  # nothing beyond the supplied window


def test_forecasters_receive_history_only_and_a_target_free_template():
    train = series(np.arange(30, dtype=float))
    folds = rolling_origin_folds(30, horizon=5, min_train_size=15, step=5)
    seen = []

    def spy(history, template, target):
        seen.append((len(history), history.index.max(), template.index.min(), list(template.columns)))
        return pd.Series(0.0, index=template.index)

    run_rolling_origin(train, "x", {"spy": spy}, folds)
    for (n_hist, last_hist, first_val, cols), f in zip(seen, folds):
        assert n_hist == f.train_size
        assert last_hist < first_val  # strictly past -> future
        assert cols == []  # no target values handed to the forecaster


def test_peak_threshold_and_scale_come_from_history_only():
    values = np.r_[np.arange(20, dtype=float), [1000.0] * 5]  # huge spike only in the future
    train = series(values)
    folds = rolling_origin_folds(25, horizon=5, min_train_size=20)
    preds = run_rolling_origin(train, "x", {"persistence": persistence_forecast}, folds)
    hist = values[:20]
    assert preds["peak_threshold"].iloc[0] == pytest.approx(np.quantile(hist, 0.9))
    assert preds["mase_scale"].iloc[0] == pytest.approx(1.0)


def test_constant_series_makes_ratio_and_mase_undefined_not_infinite():
    train = series([7.0] * 40)
    folds = rolling_origin_folds(40, horizon=10, min_train_size=20, step=10)
    s = summarize(run_rolling_origin(train, "x", SUITE, folds)).set_index("model")
    assert (s["mae"] == 0).all()
    assert s["mae_ratio_vs_persistence"].isna().all()
    assert s["mase"].isna().all()
    assert (s["folds_beating_persistence"] == 0).all()


def test_nan_actuals_are_skipped_in_scoring():
    values = np.arange(12, dtype=float)
    values[10] = np.nan  # a gap inside the validation window
    train = series(values)
    folds = rolling_origin_folds(12, horizon=4, min_train_size=8)
    preds = run_rolling_origin(train, "x", {"persistence": persistence_forecast}, folds)
    s = summarize(preds)
    assert np.isfinite(s["mae"].iloc[0])  # NaN actual dropped, rest scored


def test_non_finite_forecast_and_bad_index_are_rejected():
    train = series(np.arange(30, dtype=float))
    folds = rolling_origin_folds(30, horizon=5, min_train_size=20)
    with pytest.raises(ValueError, match="non-finite"):
        run_rolling_origin(train, "x", {"bad": lambda h, t, c: pd.Series(np.nan, index=t.index)}, folds)
    with pytest.raises(ValueError, match="index does not match"):
        run_rolling_origin(train, "x", {"bad": lambda h, t, c: pd.Series(0.0, index=t.index + 1)}, folds)


def test_folds_beyond_the_window_and_empty_folds_are_rejected():
    train = series(np.arange(20, dtype=float))
    with pytest.raises(ValueError, match="beyond"):
        run_rolling_origin(train, "x", SUITE, [RollingFold(0, 15, 15, 25)])
    with pytest.raises(ValueError, match="at least one fold"):
        run_rolling_origin(train, "x", SUITE, [])


def test_summary_without_persistence_omits_relative_columns():
    train, folds = hand_example()
    s = summarize(run_rolling_origin(train, "x", {"history_mean": mean_forecast}, folds))
    assert "mae_ratio_vs_persistence" not in s.columns


def test_error_by_horizon_is_step_by_model():
    train, folds = hand_example()
    eh = error_by_horizon(run_rolling_origin(train, "x", SUITE, folds))
    assert list(eh.index) == [1, 2]
    assert eh.loc[1, "persistence"] == pytest.approx(1.0)  # |5 - 6|
    assert eh.loc[2, "persistence"] == pytest.approx(2.0)  # |5 - 7|
    assert eh.loc[2, "history_mean"] == pytest.approx(4.5)


# ------------------------------ bands, envelope, regret ------------------------------ #

from cwfc.evaluation import band_mae, baseline_envelope, fold_mae_table, regret_summary  # noqa: E402


def test_band_mae_and_fold_table_match_hand_computation():
    train = series(range(12))  # history 0..5 at the first origin; two folds of horizon 3
    folds = rolling_origin_folds(12, horizon=3, min_train_size=6, step=3)
    preds = run_rolling_origin(train, "x", {"persistence": persistence_forecast}, folds)
    # fold 0: forecast 5 vs actual 6,7,8 -> errors 1,2,3 ; fold 1: forecast 8 vs 9,10,11 -> 1,2,3
    bands = band_mae(preds, bands=((1, 1), (2, 3)))
    assert bands.loc["steps 1-1", "persistence"] == pytest.approx(1.0)
    assert bands.loc["steps 2-3", "persistence"] == pytest.approx(2.5)
    fm = fold_mae_table(preds)
    assert fm["persistence"].tolist() == pytest.approx([2.0, 2.0])


def test_baseline_envelope_is_the_per_step_minimum():
    by_h = pd.DataFrame({"a": [1.0, 5.0, 3.0], "b": [2.0, 4.0, 3.5], "model": [0.1, 9.9, 9.9]}, index=[1, 2, 3])
    env = baseline_envelope(by_h, ["a", "b"])
    assert env.tolist() == [1.0, 4.0, 3.0]  # the model column is not a baseline


def test_regret_summary_is_relative_to_the_best_reference_in_each_fold():
    fm = pd.DataFrame(
        {"base_a": [1.0, 5.0, 3.0], "base_b": [4.0, 2.0, 3.0], "model": [2.0, 1.0, 3.5]},
        index=pd.Index([0, 1, 2], name="fold"),
    )
    r = regret_summary(fm, ["model", "base_a"], ["base_a", "base_b"]).set_index("model")
    # best reference per fold = [1, 2, 3]; model regret = [1, -1, 0.5]
    assert r.loc["model", "mean_regret"] == pytest.approx(0.5 / 3)
    assert r.loc["model", "max_regret"] == pytest.approx(1.0)
    assert r.loc["model", "folds_beating_all_references"] == 1
    # a baseline measured against the reference set including itself never has negative regret
    assert r.loc["base_a", "folds_beating_all_references"] == 0
    assert r.loc["base_a", "max_regret"] == pytest.approx(3.0)  # fold 1: 5 vs best reference 2
    assert list(r.index) == ["model", "base_a"]  # sorted by worst-case regret
