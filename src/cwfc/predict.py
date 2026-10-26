"""Produce predictions.csv for the holdout window.

Usage:
    python -m cwfc.predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cwfc.baselines import decay_forecast
from cwfc.split import split_holdout
from cwfc.targets import build_targets

logger = logging.getLogger(__name__)

TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]

def make_predictions(readings: pd.DataFrame) -> pd.DataFrame:
    target_table = build_targets(readings=readings)
    train, holdout_timestamps = split_holdout(target_table)
    template = pd.DataFrame({"timestamp": holdout_timestamps})
    predictions = {"timestamp": holdout_timestamps.to_numpy()}
    for target in TARGETS:
        fc = decay_forecast(train, template, target)
        if target == "aggregate_cpu_utilization":
            fc = fc.clip(0, 100)
        else:
            fc = fc.clip(lower=0).round().astype(int)
        predictions[target] = fc.to_numpy()

    return pd.DataFrame(predictions)
    
    
    
    


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Forecast the holdout window and write predictions.csv.")
    parser.add_argument("--data", type=Path, default=Path("challenge_data"),
                        help="folder containing vm_cpu_readings.csv.gz")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"),
                        help="where to write the predictions")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    readings = pd.read_csv(args.data / "vm_cpu_readings.csv.gz")
    predictions = make_predictions(readings)
    predictions.to_csv(args.output, index=False)
    logger.info("Wrote %d predictions to %s", len(predictions), args.output)


if __name__ == "__main__":
    main()
