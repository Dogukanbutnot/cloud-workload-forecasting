import numpy as np
import pandas as pd
import pytest

from cwfc import predict as predict_mod
from cwfc.models import damped_persistence_forecast
from cwfc.splits import split_train_holdout
from cwfc.submission import COLUMNS, validate_submission
from cwfc.targets import TARGET_CPU, clean_readings, load_cpu_readings, targets_from_cleaned
from smoke import N_TS, N_VMS, write_smoke_dataset

HORIZON = 6
TRAIN = N_TS - HORIZON
HOLDOUT_TS = [t * 300 for t in range(TRAIN, N_TS)]


def test_predict_never_reveals_the_holdout_and_meets_the_contract(tmp_path, never_reveal):
    write_smoke_dataset(tmp_path)
    frame, summary = predict_mod.predict(tmp_path, horizon=HORIZON)
    validate_submission(frame, HOLDOUT_TS)
    assert list(frame.columns) == COLUMNS and frame["timestamp"].tolist() == HOLDOUT_TS
    assert (frame["active_vm_count"] == N_VMS).all()
    assert summary["active_method"] == "fixed_population" and summary["cpu_method"] == "damped_persistence"
    assert 0 < summary["cpu_phi"] < 1


def test_cpu_forecast_equals_the_validated_model(tmp_path):
    write_smoke_dataset(tmp_path)
    frame, _ = predict_mod.predict(tmp_path, horizon=HORIZON)
    series, _ = targets_from_cleaned(clean_readings(load_cpu_readings(tmp_path)))
    split = split_train_holdout(series, horizon=HORIZON)
    expected = damped_persistence_forecast(split.train, pd.DataFrame(index=split.holdout_timestamps), TARGET_CPU)
    assert frame[TARGET_CPU].to_numpy() == pytest.approx(expected.to_numpy())


def test_forecast_decays_from_the_last_value_toward_the_history_mean(tmp_path):
    write_smoke_dataset(tmp_path)
    frame, s = predict_mod.predict(tmp_path, horizon=HORIZON)
    f = frame[TARGET_CPU].to_numpy()
    dist = np.abs(f - s["cpu_history_mean"])
    assert (np.diff(dist) <= 1e-12).all()  # monotone convergence to the mean


def test_prediction_is_deterministic(tmp_path):
    write_smoke_dataset(tmp_path)
    a, _ = predict_mod.predict(tmp_path, horizon=HORIZON)
    b, _ = predict_mod.predict(tmp_path, horizon=HORIZON)
    pd.testing.assert_frame_equal(a, b)


def test_population_not_proven_fixed_falls_back_to_persistence(tmp_path):
    write_smoke_dataset(tmp_path, delete_vm0_at=(N_TS - 2) * 300)  # a VM is deleted inside the horizon
    frame, summary = predict_mod.predict(tmp_path, horizon=HORIZON)
    assert summary["active_method"].startswith("persistence_fallback")
    assert (frame["active_vm_count"] == N_VMS).all()  # last observed count, carried forward
    validate_submission(frame, HOLDOUT_TS)


def test_missing_metadata_also_falls_back(tmp_path):
    write_smoke_dataset(tmp_path)
    (tmp_path / "vm_metadata.csv.gz").unlink()
    _, summary = predict_mod.predict(tmp_path, horizon=HORIZON)
    assert summary["active_method"].startswith("persistence_fallback")
    assert "population_verdict" not in summary


def test_very_short_history_falls_back_to_persistence_for_cpu_too(tmp_path):
    write_smoke_dataset(tmp_path)
    frame, summary = predict_mod.predict(tmp_path, horizon=N_TS - 8)  # only 8 training points
    assert summary["cpu_method"] == "persistence_fallback_short_history"
    assert frame[TARGET_CPU].nunique() == 1  # flat last value


def test_training_gap_is_refused_instead_of_invented(tmp_path):
    write_smoke_dataset(tmp_path)
    p = tmp_path / "vm_cpu_readings.csv.gz"
    df = pd.read_csv(p)
    df[df["timestamp"] != 5 * 300].to_csv(p, index=False)  # a whole timestamp vanishes in the training window
    with pytest.raises(ValueError, match="gap timestamps"):
        predict_mod.predict(tmp_path, horizon=HORIZON)


def test_cli_writes_a_valid_file_without_index(tmp_path, capsys):
    write_smoke_dataset(tmp_path)
    out = tmp_path / "out" / "predictions.csv"
    predict_mod.main(["--data", str(tmp_path), "--output", str(out), "--horizon", str(HORIZON)])
    back = pd.read_csv(out)
    assert list(back.columns) == COLUMNS and len(back) == HORIZON
    assert out.read_text().splitlines()[0] == ",".join(COLUMNS)
    assert '"rows": 6' in capsys.readouterr().out


def test_data_argument_may_be_the_readings_file_itself(tmp_path):
    write_smoke_dataset(tmp_path)
    from_dir, s_dir = predict_mod.predict(tmp_path, horizon=HORIZON)
    from_file, s_file = predict_mod.predict(tmp_path / "vm_cpu_readings.csv.gz", horizon=HORIZON)
    pd.testing.assert_frame_equal(from_dir, from_file)
    assert s_file["active_method"] == "fixed_population"  # metadata was found next to the file
