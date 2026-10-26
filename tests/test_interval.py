import numpy as np
import pandas as pd
import pytest

from cwfc.interval import add_interval, residual_quantiles


def make_results(residuals_per_step):
    """Fake backtest output: forecast 0, so actual == residual."""
    rows = [
        {"fold": fold, "step": step, "actual": r, "forecast": 0.0}
        for step, values in residuals_per_step.items()
        for fold, r in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_one_row_per_step_and_shared_within_band():
    rng = np.random.default_rng(0)
    results = make_results({step: rng.normal(size=7) for step in range(1, 25)})

    q = residual_quantiles(results)

    assert q["step"].tolist() == list(range(1, 25))
    assert q.loc[q["step"].between(1, 6), "lower"].nunique() == 1
    assert (q["lower"] < q["upper"]).all()


def test_band_quantiles_reflect_band_spread():
    # steps 1-6 have small errors, steps 13-24 large ones
    results = make_results(
        {step: np.linspace(-1, 1, 7) * (1 if step <= 6 else 2 if step <= 12 else 5) for step in range(1, 25)}
    )

    q = residual_quantiles(results).set_index("step")

    assert q.loc[1, "upper"] < q.loc[7, "upper"] < q.loc[13, "upper"]


def test_invalid_coverage_raises():
    results = make_results({step: [0.0] * 7 for step in range(1, 25)})
    with pytest.raises(ValueError, match="coverage"):
        residual_quantiles(results, coverage=1.5)


def test_add_interval_aligns_by_position_not_index():
    # the forecast index is the template index (e.g. 110..133), quantiles start at 0
    forecast = pd.Series([10.0, 11.0], index=[110, 111])
    quantiles = pd.DataFrame({"step": [1, 2], "lower": [-1.0, -2.0], "upper": [0.5, 1.5]})

    table = add_interval(forecast, quantiles)

    assert table["lower"].tolist() == [9.0, 9.0]
    assert table["upper"].tolist() == [10.5, 12.5]
    assert table["forecast"].tolist() == [10.0, 11.0]
