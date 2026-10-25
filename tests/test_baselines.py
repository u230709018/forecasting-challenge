import pandas as pd

from cwfc.baselines import decay_forecast, mean_forecast


def make_train(values):
    return pd.DataFrame({"y": values})


TEMPLATE = pd.DataFrame(index=range(100, 124))  # 24 forecast steps


def test_mean_forecast_full_and_window():
    train = make_train([1.0, 2.0, 3.0, 10.0])

    assert mean_forecast(train, TEMPLATE, "y").tolist() == [4.0] * 24
    assert mean_forecast(train, TEMPLATE, "y", window=2).tolist() == [6.5] * 24


def test_decay_forecast_on_constant_series_is_finite_and_constant():
    # a constant series has undefined autocorrelation (NaN); the forecast must stay finite
    forecast = decay_forecast(make_train([10000.0] * 50), TEMPLATE, "y")

    assert forecast.notna().all()
    assert forecast.tolist() == [10000.0] * 24


def test_decay_forecast_moves_from_last_value_towards_mean():
    train = make_train([9.0, 9.2, 9.1, 9.3, 9.6, 9.9])
    mean, last = train["y"].mean(), train["y"].iloc[-1]

    forecast = decay_forecast(train, TEMPLATE, "y")

    assert mean < forecast.iloc[0] < last          # first step between mean and last value
    assert forecast.is_monotonic_decreasing        # moves steadily towards the mean
    assert abs(forecast.iloc[-1] - mean) < abs(forecast.iloc[0] - mean)


def test_decay_forecast_with_negative_autocorrelation_falls_back_to_mean():
    train = make_train([1.0, 3.0] * 10)  # alternating series: lag-1 autocorrelation is negative

    forecast = decay_forecast(train, TEMPLATE, "y")

    assert forecast.tolist() == [train["y"].mean()] * 24
