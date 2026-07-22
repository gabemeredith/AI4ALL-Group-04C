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

`nasa_cmapss_FD001_scaled.csv` (committed) is the **single source of truth** — everything
(baseline, trainer, and all three dashboard modes) reads it. 20,631 rows, 100 engines.
Columns: `unit_number`, `time_in_cycles`, `operational_setting_1`,
`sensor_2/3/4/6/7/8/9/11/12/13/14/15/17/20/21`, `RUL`. That gives the model 16 features
(`operational_setting_1` + those 15 sensors); they're listed as `FEATURE_COLS` in `model.py`.

Note `RUL` in the file is **raw/uncapped** (0–361). Both the trainer and the dashboard apply
the piecewise-linear cap at 125 cycles themselves, so don't assume the on-disk RUL is capped.

Historical note: a second, differently-preprocessed `train_FD001_scaled.csv` (schema `unit`,
`cycle`, `op_setting_1`, `op_setting_2`, 14 sensors — no `sensor_6`) circulated on the team
earlier and the code used to target it. We deliberately consolidated onto the committed file
so the project trains and runs with no external file dependency; the schemas are **not**
interchangeable. If the team later standardizes on that other file, realign `FEATURE_COLS`
and the loaders in `save_model.py`/`app.py`.

All sensor/operational-setting columns are already z-score scaled — values are not in raw
physical units. Keep that in mind in any UI copy ("scaled sensor reading", not "°F"/"psi").

## Models

Two models are planned per the team deck:
- **Logistic Regression** — interpretable baseline; classifies "will this engine fail within
  W cycles?"
- **Random Forest** — captures nonlinear degradation and sensor interactions

`logistic_regression_base.py` is the existing baseline script: it frames the same problem two
ways (logistic classification vs. linear regression on capped RUL) on an identical
engine-grouped train/test split, and reports both native metrics and a shared classification
scoreboard, plus a custom asymmetric C-MAPSS scoring function.

`save_model.py` trains the real `RandomForestRegressor` for the dashboard on
`nasa_cmapss_FD001_scaled.csv`, and saves it to `random_forest_model.pkl` (not committed —
generate it locally with `python save_model.py`). It caps RUL at 125, evaluates on an
engine-grouped hold-out (RMSE ≈ 18.0, MAE ≈ 13.3, R² ≈ 0.81), prints feature importances,
then refits on all 100 engines before saving. The RF is tuned (`min_samples_leaf=5`,
`max_features="sqrt"`, `n_estimators=400`) — on this data/feature set tree models top out
around RMSE 18; rolling-window features and gradient boosting were tried and didn't beat it.

No LSTM model exists in this repo. `model.py`'s `predict_rul()` loads
`random_forest_model.pkl` when present; when it isn't, it falls back to a placeholder
heuristic (clearly marked, not a trained model) so the dashboard still shows something. That
function is the only place a real/updated model needs to be wired in.

## Dashboard

`app.py` — a Streamlit app with three sidebar-selectable modes, all backed by the same
`predict_rul()`:
- **Browse engine** — pick an engine from `nasa_cmapss_FD001_scaled.csv`, scrub cycles, see
  sensor charts and predicted RUL. (On load the app renames `unit_number`/`time_in_cycles` to
  `unit`/`cycle` internally.)
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
