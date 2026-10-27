# Cloud Workload Forecasting — Submission

Forecasts the final 2 hours (24 five-minute steps) of `active_vm_count` and `aggregate_cpu_utilization` for a curated 10,000-VM subset of the Microsoft Azure Public Dataset V2.

- **Final report:** [`reports/final_report.md`](reports/final_report.md)
- **Predictions:** [`predictions.csv`](predictions.csv)
- Original challenge: [`ASSIGNMENT.md`](ASSIGNMENT.md), [`DATASET.md`](DATASET.md), [`SUBMISSION.md`](SUBMISSION.md)

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
```

On Windows, if `python` opens the Microsoft Store, create the environment with `py -3.12 -m venv .venv` (or your installed 3.11+ version).

### Dataset

Download `data-v1.tar.gz` from the [data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract it into `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data && sha256sum -c SHA256SUMS && cd ..
```

`challenge_data/` is git-ignored; only its README is tracked.

### Generate `predictions.csv` (one command)

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

`--data` is any folder containing `vm_cpu_readings.csv.gz`. The command builds the target series, splits off the last 24 timestamps as the holdout, fits the final model on the rest and writes 24 rows with the columns `timestamp,active_vm_count,aggregate_cpu_utilization`. There is no separate training step: the model is fitted inside this command.

### Tests

```bash
python -m pytest -q
```

23 tests, including an end-to-end smoke test that runs the CLI on a tiny dataset with churn, a missing value and an out-of-range value.

## Repository layout

| Path | Contents |
|---|---|
| `src/cwfc/targets.py` | `build_targets`: validity rule, per-timestamp VM count and mean CPU |
| `src/cwfc/split.py` | `split_holdout`: the only code that touches the holdout; returns its timestamps only |
| `src/cwfc/baselines.py` | persistence, mean (optionally windowed) and the final **decay** model |
| `src/cwfc/evaluation.py` | `backtest`: rolling-origin evaluation (7 folds) with an empty-fold guard |
| `src/cwfc/interval.py` | empirical prediction intervals from backtest residuals |
| `src/cwfc/predict.py` | CLI that writes `predictions.csv` |
| `src/cwfc/metrics.py`, `validation.py` | starter utilities (metrics, rolling-origin splits) |
| `notebooks/01_data_audit.ipynb` | structure, coverage, quality, metadata, targets |
| `notebooks/02_eda.ipynb` | trend, bump drivers, autocorrelation, per-VM distribution, segments |
| `notebooks/03_baselines.ipynb` | baselines and decay on the rolling-origin folds |
| `notebooks/04_holdout_evaluation.ipynb` | the single look at the holdout |
| `notebooks/05_capacity.ipynb` | capacity margin trade-off |
| `reports/` | final report and figures |
| `tests/` | unit and smoke tests |

Run the notebooks from the `notebooks/` folder in numeric order; they read data from `../challenge_data/`.

## Main modelling decisions

- **Targets:** a reading is valid if `timestamp`, `vm_id` and `avg_cpu` are present and `avg_cpu` is in [0, 100]; invalid rows are dropped (and counted in the log), never imputed. `active_vm_count` counts distinct VMs with a valid reading; `aggregate_cpu_utilization` is their mean `avg_cpu`.
- **Leakage:** the holdout values are reachable only through `split_holdout`, which hands out timestamps only. No calendar features (the wall-clock time of 0 is unknown) and no lifetime metadata features (they summarise the holdout period too).
- **Validation:** rolling-origin inside the 110-step training window, minimum training size 50, step 6, horizon 24 → 7 folds. The starter `rolling_origin_splits` defaults (1-day horizon, 7-day minimum) return no folds on this series; parameters are always explicit.
- **Model:** decay, `forecast(h) = mean + (last − mean) × φ^h` with φ the lag-1 autocorrelation of the training data (0 if undefined or negative). Fold MAE 0.201 vs 0.215 for persistence and 0.209 for the mean. Counts are rounded to whole VMs; CPU is clipped to [0, 100].
- **Capacity:** provision forecast + the q90 residual margin per horizon band (about 3–4% of the current level); q95 for Interactive workloads; a peak-risk trigger when the latest value exceeds the forecast. See section 8 of the report.

## Use of AI assistance

This project was developed with AI assistance (Claude) for implementation and drafting. Problem framing, design decisions, analysis and the final model are my own, and I reviewed every part of the code and the report.

## Source

Microsoft Azure Public Dataset V2: https://github.com/Azure/AzurePublicDataset
