import logging

import numpy as np
import pandas as pd
import pytest

from cwfc.targets import build_targets


def test_invalid_rows_are_dropped():
    readings = pd.DataFrame({
        "timestamp": [0, 0, 0, 300, 300, 300, 300],
        "vm_id": ["A", "B", "C", "A", "B", "C", "D"],
        "avg_cpu": [10, 20, 30, 12, 18, np.nan, 150]
    })

    result = build_targets(readings=readings)

    expected = pd.DataFrame({
        "timestamp": [0, 300],
        "active_vm_count": [3, 2],
        "aggregate_cpu_utilization": [20.0, 15.0]
    })

    pd.testing.assert_frame_equal(result, expected)


def test_missing_column_raises():
    readings = pd.DataFrame({
        "timestamp": [0, 0, 300],
        "vm_id": ["A", "B", "C"]
    })

    with pytest.raises(ValueError, match="avg_cpu"):
        build_targets(readings=readings)


def test_invalid_row_warns(caplog):
    readings = pd.DataFrame({
        "timestamp": [0, 0, 0, 300, 300, 300, 300],
        "vm_id": ["A", "B", "C", "A", "B", "C", "D"],
        "avg_cpu": [10, 20, 30, 12, 18, np.nan, 150]
    })
    with caplog.at_level(logging.INFO):                                  
        build_targets(readings=readings)
    levels = [r.levelname for r in caplog.records]
    assert "WARNING" in levels


def test_clean_rows_do_not_warn(caplog):
    readings = pd.DataFrame({
        "timestamp": [0, 0, 0, 300, 300, 300, 300],
        "vm_id": ["A", "B", "C", "A", "B", "C", "D"],
        "avg_cpu": [10, 20, 30, 12, 18, 50, 100]
    })                                   
    with caplog.at_level(logging.INFO):
        build_targets(readings=readings)
    levels = [r.levelname for r in caplog.records]
    assert "WARNING" not in levels