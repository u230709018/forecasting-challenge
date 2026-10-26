from __future__ import annotations

import pandas as pd

COVERAGE = 0.8
# Horizon steps are pooled into bands: with 7 folds there are only 7 residuals per single step.
STEP_BANDS = [(1, 6), (7, 12), (13, 24)]


def residual_quantiles(
    results: pd.DataFrame,
    coverage: float = COVERAGE,
    bands: list[tuple[int, int]] = STEP_BANDS,
) -> pd.DataFrame:
    """Empirical interval offsets per horizon step from backtest residuals.

    `results` is the output of `backtest`. Returns one row per step with the
    `lower` and `upper` residual quantiles of the band that step belongs to.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    residual = results["actual"] - results["forecast"]
    tail = (1 - coverage) / 2

    rows = []
    for start, end in bands:
        band = residual[results["step"].between(start, end)]
        if band.empty:
            raise ValueError(f"No residuals for steps {start}-{end}")
        lower, upper = band.quantile(tail), band.quantile(1 - tail)
        rows += [{"step": s, "lower": lower, "upper": upper} for s in range(start, end + 1)]
    return pd.DataFrame(rows)



def add_interval(forecast: pd.Series, quantiles: pd.DataFrame) -> pd.DataFrame:
    values = forecast.to_numpy()
    return pd.DataFrame({
        "forecast": values,
        "lower": values + quantiles["lower"].to_numpy(),
        "upper": values + quantiles["upper"].to_numpy()
    })