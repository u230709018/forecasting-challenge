from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {"timestamp", "vm_id", "avg_cpu"}

def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(readings.columns)
    if missing_columns:
        raise ValueError(f"A required column is missing. {sorted(missing_columns)}")

    not_null = readings[sorted(REQUIRED_COLUMNS)].notna().all(axis=1)
    in_range = readings["avg_cpu"].between(0, 100)

    mask = not_null & in_range
    valid = readings[mask]

    dropped_rows = len(readings) - len(valid)
    if dropped_rows == 0:
        logger.info("Dropped %d of %d rows", dropped_rows, len(readings))
    else:
        logger.warning("Dropped %d of %d rows", dropped_rows, len(readings))
    targets = valid.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
        ).reset_index()

    return targets