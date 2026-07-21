# AI4ALL Group 04C — Predicting Jet Engine Failure

Predicting a turbofan engine's Remaining Useful Life (RUL) — how many flight cycles remain
before failure — from sensor telemetry, to support predictive maintenance instead of fixed
replacement schedules.

## Dataset

`nasa_cmapss_FD001_scaled.csv` — NASA C-MAPSS FD001 turbofan degradation dataset (Prognostics
CoE, NASA Ames): [kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).
100 engines, sensor and operational-setting readings already z-score scaled, with ground-truth
RUL per row.

## Setup

```
pip install -r requirements.txt
```

## Baseline model

```
python logistic_regression_base.py
```

Compares a logistic regression classifier ("will this engine fail within W cycles?") against
a linear regression on capped RUL, evaluated on an identical engine-grouped train/test split.

## Dashboard

```
streamlit run app.py
```

Three modes, picked from the sidebar:
- **Browse engine** — pick a training-set engine, scrub through its cycles, see its sensor
  readings and predicted RUL. Requires `train_FD001_scaled.csv` in the repo root (see below).
- **Upload CSV** — upload sensor readings and get predicted RUL per row, with a CSV download
  of the results.
- **Manual input** — type in one engine's sensor readings by hand and get a predicted RUL.

All three call the same `predict_rul()` in `model.py`.

## Trained model

`save_model.py` trains a `RandomForestRegressor` on `train_FD001_scaled.csv` and saves it to
`random_forest_model.pkl`:

```
python save_model.py
```

`train_FD001_scaled.csv` isn't committed yet — get it from a teammate and drop it in the repo
root before running this. Until `random_forest_model.pkl` exists, the dashboard falls back to
a placeholder heuristic in `model.py` (clearly not a trained model, just enough to keep the UI
functional) — the sidebar shows a warning when this fallback is active.
