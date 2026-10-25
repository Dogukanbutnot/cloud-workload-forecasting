"""Tiny synthetic stand-in for the challenge data (a 'smoke' dataset)."""

import numpy as np
import pandas as pd

N_VMS, N_TS = 20, 60
CATEGORIES = ["Delay-insensitive", "Interactive", "Unknown"]


def write_smoke_dataset(path, delete_vm0_at=None):
    rng = np.random.default_rng(0)
    t = np.arange(N_TS)
    frames = []
    for i in range(N_VMS):
        avg = np.clip(20 + 5 * np.sin(t / 5) + rng.normal(0, 1, N_TS) + i * 0.1, 0, 100)
        frames.append(
            pd.DataFrame(
                {"timestamp": t * 300, "vm_id": f"vm{i}", "min_cpu": avg / 2,
                 "max_cpu": np.clip(avg * 2, 0, 100), "avg_cpu": avg}
            )
        )
    pd.concat(frames).to_csv(path / "vm_cpu_readings.csv.gz", index=False)
    deleted = [10**9] * N_VMS
    if delete_vm0_at is not None:
        deleted[0] = delete_vm0_at
    pd.DataFrame(
        {
            "vm_id": [f"vm{i}" for i in range(N_VMS)],
            "subscription_id": [f"sub{i % 7}" for i in range(N_VMS)],
            "deployment_id": [f"dep{i % 9}" for i in range(N_VMS)],
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": deleted,
            "max_cpu": [90.0 - i for i in range(N_VMS)], "avg_cpu": [15.0 + i * 0.3 for i in range(N_VMS)],
            "p95_max_cpu": [60.0 - i for i in range(N_VMS)],
            "vm_category": [CATEGORIES[i % 3] for i in range(N_VMS)],
            "virtual_core_bucket": ["2" if i % 4 else "4" for i in range(N_VMS)],
            "memory_gb_bucket": ["4" if i % 2 else "8" for i in range(N_VMS)],
        }
    ).to_csv(path / "vm_metadata.csv.gz", index=False)
