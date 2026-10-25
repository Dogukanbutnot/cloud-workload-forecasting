import json

import pytest

from cwfc import run_eda
from smoke import N_TS, N_VMS, write_smoke_dataset

HORIZON = 6
TRAIN = N_TS - HORIZON


def test_eda_uses_only_the_training_window_and_never_reveals_the_holdout(tmp_path, never_reveal):
    write_smoke_dataset(tmp_path)
    res = run_eda.run(tmp_path, horizon=HORIZON, output_dir=tmp_path / "eda", figures=False)
    c = res["summary"]["coverage"]
    assert c["window_end"] == (TRAIN - 1) * 300  # last training timestamp, not the last timestamp
    assert c["n_timestamps"] == TRAIN and c["n_rows"] == N_VMS * TRAIN  # holdout rows never entered
    assert c["n_vms_full_coverage"] == N_VMS and c["n_gap_timestamps"] == 0


def test_summary_json_and_tables_are_written_and_complete(tmp_path):
    write_smoke_dataset(tmp_path)
    res = run_eda.run(tmp_path, horizon=HORIZON, output_dir=tmp_path / "eda", figures=False)
    out = tmp_path / "eda"
    payload = json.loads((out / "eda_summary.json").read_text())
    for key in ("coverage", "distribution", "variance_structure", "temporal", "peak_low",
                "volatility_persistence", "burstiness", "lifetimes", "tenants",
                "metadata_representativeness", "candidate_targets"):
        assert key in payload
    assert (out / "candidate_targets.csv").exists()
    for col in ("vm_category", "virtual_core_bucket", "memory_gb_bucket"):
        assert (out / f"segments_{col}.csv").exists()
        assert not res["tables"][f"segments_{col}"].empty
    assert res["figures"] == []


def test_segment_levels_and_lifetimes_come_from_metadata(tmp_path):
    write_smoke_dataset(tmp_path)
    res = run_eda.run(tmp_path, horizon=HORIZON)
    seg = res["tables"]["segments_vm_category"]
    assert set(seg["segment"]) == {"Delay-insensitive", "Interactive", "Unknown"}
    assert seg["n_vms"].sum() == N_VMS
    assert res["summary"]["lifetimes"]["share_created_at_trace_start"] == 1.0


def test_findings_digest_mentions_the_key_numbers(tmp_path):
    write_smoke_dataset(tmp_path)
    lines = run_eda.findings(run_eda.run(tmp_path, horizon=HORIZON)["summary"])
    text = "\n".join(lines)
    assert f"{N_VMS} VMs" in text and "effective independent VMs" in text and "Lifetimes" in text


def test_figures_are_created_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    write_smoke_dataset(tmp_path)
    res = run_eda.run(tmp_path, horizon=HORIZON, output_dir=tmp_path / "eda", figures=True)
    assert len(res["figures"]) == 6
    for p in res["figures"]:
        assert (tmp_path / "eda" / p.split("/")[-1]).stat().st_size > 1000


def test_no_metadata_still_runs_the_target_only_analysis(tmp_path):
    write_smoke_dataset(tmp_path)
    (tmp_path / "vm_metadata.csv.gz").unlink()
    res = run_eda.run(tmp_path, horizon=HORIZON)
    assert "lifetimes" not in res["summary"] and "temporal" in res["summary"]
