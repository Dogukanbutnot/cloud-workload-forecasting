import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cwfc
from cwfc import evaluate_holdout, predict as predict_mod
from cwfc.submission import write_submission
from cwfc.targets import TARGET_ACTIVE, TARGET_CPU
from smoke import N_TS, write_smoke_dataset

HORIZON = 6
KW = dict(horizon=HORIZON, min_train_size=30, step=6)


def prepare(tmp_path):
    write_smoke_dataset(tmp_path)
    frame, _ = predict_mod.predict(tmp_path, horizon=HORIZON)
    path = tmp_path / "predictions.csv"
    write_submission(frame, path, frame["timestamp"])
    return frame, path


def test_reveal_holdout_is_called_in_exactly_one_module():
    src = Path(cwfc.__file__).parent
    callers = {p.name for p in src.glob("*.py") if re.search(r"\.reveal_holdout\(", p.read_text())}
    assert callers == {"evaluate_holdout.py"}


def test_structure_and_metrics_on_the_smoke_holdout(tmp_path):
    _, path = prepare(tmp_path)
    r = evaluate_holdout.run(tmp_path, path, output_dir=tmp_path / "out", **KW)
    assert r["n_holdout"] == HORIZON
    cpu = r["targets"][TARGET_CPU]
    for k in ("mae", "rmse", "smape_pct", "bias", "mase"):
        assert k in cpu["submission"]
    assert cpu["submission"]["rmse"] >= cpu["submission"]["mae"] >= 0
    assert set(cpu["details"]["mae_by_band"]) == {"steps 1-6"}  # a 6-step horizon has no later bands
    assert "policy" in cpu["details"] and 0 <= cpu["details"]["policy"]["holdout_with_margin"]["under_rate"] <= 1
    assert r["validation_fold_mae_damped_persistence"]["min"] <= r["validation_fold_mae_damped_persistence"]["max"]
    # the constant-population target is exact on this dataset, and ratios against an exact reference are undefined
    active = r["targets"][TARGET_ACTIVE]
    assert active["submission"]["mae"] == 0.0 and np.isnan(active["submission"]["mae_ratio_vs_persistence"])
    for name in ("holdout_metrics.json", "holdout_forecast_vs_actual.csv"):
        assert (tmp_path / "out" / name).exists()


def test_the_submitted_file_is_what_gets_scored(tmp_path):
    frame, path = prepare(tmp_path)
    base = evaluate_holdout.run(tmp_path, path, output_dir=tmp_path / "out", **KW)
    tampered = frame.assign(**{TARGET_CPU: 50.0})
    bad_path = tmp_path / "tampered.csv"
    write_submission(tampered, bad_path, tampered["timestamp"])
    r = evaluate_holdout.run(tmp_path, bad_path, output_dir=tmp_path / "out2", **KW)
    actual = pd.read_csv(tmp_path / "out2" / "holdout_forecast_vs_actual.csv")[f"{TARGET_CPU}_actual"]
    assert r["targets"][TARGET_CPU]["submission"]["mae"] == pytest.approx(np.abs(50.0 - actual).mean())
    assert r["targets"][TARGET_CPU]["submission"]["mae"] > base["targets"][TARGET_CPU]["submission"]["mae"]
    # the references are built from the training window only, so they do not move
    assert r["targets"][TARGET_CPU]["persistence"] == base["targets"][TARGET_CPU]["persistence"]


def test_invalid_predictions_are_rejected_before_anything_is_revealed(tmp_path, never_reveal):
    frame, path = prepare(tmp_path)
    frame.iloc[:3].to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match="invalid submission"):
        evaluate_holdout.run(tmp_path, tmp_path / "short.csv", **KW)  # never_reveal would raise AssertionError otherwise


def test_cli_refuses_to_reveal_without_confirmation(tmp_path):
    _, path = prepare(tmp_path)
    with pytest.raises(SystemExit, match="--confirm"):
        evaluate_holdout.main(["--data", str(tmp_path), "--predictions", str(path), "--horizon", str(HORIZON)])


def test_cli_with_confirmation_prints_json(tmp_path, capsys):
    _, path = prepare(tmp_path)
    evaluate_holdout.main(["--data", str(tmp_path), "--predictions", str(path), "--horizon", str(HORIZON),
                           "--min-train-size", "30", "--step", "6", "--confirm"])
    assert '"n_holdout": 6' in capsys.readouterr().out
