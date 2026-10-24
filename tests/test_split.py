from __future__ import annotations

import pandas as pd
import pytest

from cwfc.split import split_holdout, HOLDOUT_SIZE

def test_split_returns_train_and_last_timestamps():
    table = pd.DataFrame({"timestamp": [i * 300 for i in range(30)], "active_vm_count": [10] * 30, "aggregate_cpu_utilization": [9.0] * 30})

    train, holdout = split_holdout(table=table, holdout_size=24)
    
    assert len(train) == 6
    assert holdout.tolist() == [i * 300 for i in range(6, 30)]
    assert holdout.name == "timestamp"


def test_split_short_table_raises():
    table = pd.DataFrame({"timestamp": [i * 300 for i in range(24)], "active_vm_count": [10] * 24, "aggregate_cpu_utilization": [9.0] * 24})

    with pytest.raises(ValueError, match="greater than 24"):
        split_holdout(table)


def test_split_unsorted_raises():
    table = pd.DataFrame({
        "timestamp": [i * 300 for i in range(30)],
        "active_vm_count": [10] * 30,
        "aggregate_cpu_utilization": [9.0] * 30,
    }).iloc[::-1]

    with pytest.raises(ValueError, match="sorted"):
        split_holdout(table)


def test_split_duplicates_raises():
    table = pd.DataFrame({
        "timestamp": [i * 300 for i in range(29)] + [28 * 300],
        "active_vm_count": [10] * 30,
        "aggregate_cpu_utilization": [9.0] * 30,
    })

    with pytest.raises(ValueError, match="unique"):
        split_holdout(table)