import pandas as pd
import pytest

from cwfc.baselines import persistence_forecast
from cwfc.evaluation import backtest


def make_train(n, values=None):
    return pd.DataFrame({
        "timestamp": [i * 300 for i in range(n)],
        "aggregate_cpu_utilization": values if values is not None else [9.0] * n,
    })


def test_backtest_folds_on_training_window():
    results = backtest(make_train(110), persistence_forecast, "aggregate_cpu_utilization")

    # (110 - 50 - 24) / 6 + 1 = 7 folds, each with 24 steps
    assert results["fold"].nunique() == 7
    assert results.groupby("fold").size().tolist() == [24] * 7
    # the last fold ends exactly at the end of the training window
    assert results["timestamp"].max() == 109 * 300


def test_backtest_raises_when_no_folds():
    with pytest.raises(ValueError, match="No folds"):
        backtest(make_train(60), persistence_forecast, "aggregate_cpu_utilization")


def test_persistence_on_constant_series_has_zero_error():
    results = backtest(make_train(110), persistence_forecast, "aggregate_cpu_utilization")

    assert (results["actual"] - results["forecast"]).abs().max() == 0
