# AI4ALL-Group-04C

## Project

"Predicting Jet Engine Failure Through Machine Learning" — an AI4ALL course group project.
Team: Erronn Bridgewater, Tom Chatto, Gabe Meredith, Nish Methuku, Alejandro Hernandez, William Mahunda, Hunter Ngo.

Goal: predict a turbofan engine's Remaining Useful Life (RUL) — how many flight cycles remain
before failure — from sensor telemetry, to support predictive maintenance instead of fixed
replacement schedules.

Research question: can ML models predict early engine failure to support predictive
maintenance and reduce unexpected downtime?

Full background (problem motivation, bias/mitigation notes, citations) lives in the team's
project deck, not duplicated here.

## Datasets

Two differently-preprocessed versions of the same underlying NASA C-MAPSS FD001 data exist
across the team — don't assume they're interchangeable:

- `nasa_cmapss_FD001_scaled.csv` (committed) — 20,631 rows, 100 engines. Columns:
  `unit_number`, `time_in_cycles`, `operational_setting_1`, `sensor_2/3/4/6/7/8/9/11/12/13/14/15/17/20/21`,
  `RUL`. Used only by `logistic_regression_base.py`.
- `train_FD001_scaled.csv` (**not committed yet** — get it from a teammate and drop it in the
  repo root) — expected columns: `unit`, `cycle`, `op_setting_1`, `op_setting_2`, and 14 named
  sensors (see `FEATURE_COLS` in `model.py`; notably excludes `sensor_6` but includes
  `op_setting_2`, unlike the file above). Used by `save_model.py` and the dashboard's
  "Browse engine" mode.

All sensor/operational-setting columns in both files are already z-score scaled — values are
not in raw physical units. Keep that in mind in any UI copy ("scaled sensor reading", not
"°F"/"psi").

## Models

Two models are planned per the team deck:
- **Logistic Regression** — interpretable baseline; classifies "will this engine fail within
  W cycles?"
- **Random Forest** — captures nonlinear degradation and sensor interactions

`logistic_regression_base.py` is the existing baseline script: it frames the same problem two
ways (logistic classification vs. linear regression on capped RUL) on an identical
engine-grouped train/test split, and reports both native metrics and a shared classification
scoreboard, plus a custom asymmetric C-MAPSS scoring function.

`save_model.py` trains the real `RandomForestRegressor` for the dashboard, on
`train_FD001_scaled.csv`, and saves it to `random_forest_model.pkl` (also not committed —
generate it locally by running the script once that CSV is in place).

No LSTM model exists in this repo. `model.py`'s `predict_rul()` loads
`random_forest_model.pkl` when present; when it isn't, it falls back to a placeholder
heuristic (clearly marked, not a trained model) so the dashboard still shows something. That
function is the only place a real/updated model needs to be wired in.

## Dashboard

`app.py` — a Streamlit app with three sidebar-selectable modes, all backed by the same
`predict_rul()`:
- **Browse engine** — pick an engine from `train_FD001_scaled.csv`, scrub cycles, see sensor
  charts and predicted RUL (disabled with a clear message if that CSV isn't present).
- **Upload CSV** — upload sensor readings, get predicted RUL per row, download results.
- **Manual input** — hand-enter one engine's sensor readings for a single prediction.

This is a merge of two dashboards that existed briefly in parallel (one engine-browser-first,
one upload/manual-input-first) — there is only one dashboard entry point now, `app.py`.

Run it with:
```
pip install -r requirements.txt
streamlit run app.py
```

## Conventions

- Do not add a `Co-Authored-By: Claude` trailer to commits in this repo.
- Keep commits scoped and descriptive — this is a team repo other members read the history of.
