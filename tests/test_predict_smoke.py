"""End-to-end smoke test: run the CLI on a tiny, messy dataset like a reviewer would."""
import numpy as np
import pandas as pd

from cwfc.predict import main

N_STEPS = 40  # 16 training steps + 24 holdout steps
VMS = ["vm-a", "vm-b", "vm-c", "vm-d", "vm-e"]


def write_tiny_dataset(folder):
    rng = np.random.default_rng(42)
    rows = [
        {"timestamp": t * 300, "vm_id": vm, "min_cpu": 0.0, "max_cpu": 100.0, "avg_cpu": rng.uniform(1, 60)}
        for t in range(N_STEPS)
        for vm in VMS
        if not (vm == "vm-e" and t >= 20)  # churn: vm-e is deleted halfway
    ]
    readings = pd.DataFrame(rows)
    readings.loc[3, "avg_cpu"] = np.nan   # missing value
    readings.loc[7, "avg_cpu"] = 150.0    # out-of-range value
    readings.to_csv(folder / "vm_cpu_readings.csv.gz", index=False)
    return readings


def test_cli_produces_valid_predictions_on_tiny_messy_data(tmp_path):
    readings = write_tiny_dataset(tmp_path)
    output = tmp_path / "predictions.csv"

    main(["--data", str(tmp_path), "--output", str(output)])

    predictions = pd.read_csv(output)
    holdout = sorted(readings["timestamp"].unique())[-24:]

    assert list(predictions.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert predictions["timestamp"].tolist() == holdout
    values = predictions[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy()
    assert np.isfinite(values).all()
    assert (predictions["active_vm_count"] >= 0).all()
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()
