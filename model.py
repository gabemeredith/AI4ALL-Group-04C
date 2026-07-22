"""
RUL prediction.

Loads the trained Random Forest (random_forest_model.pkl, produced by
save_model.py) if it exists. Falls back to a placeholder heuristic — NOT a
trained model — so the dashboard still shows something before the pickle
exists. Run `python save_model.py` to train and save the real model on
nasa_cmapss_FD001_scaled.csv (committed to the repo).
"""

import os

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "random_forest_model.pkl"

# The 16 model inputs, matching the columns of nasa_cmapss_FD001_scaled.csv
# (the single committed, labeled FD001 dataset). All are z-score scaled.
FEATURE_COLS = [
    "operational_setting_1",
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

RUL_CAP = 125


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_rul(rows: pd.DataFrame, model=None) -> np.ndarray:
    """Predict RUL for rows containing FEATURE_COLS.

    Uses `model.predict()` when a trained model is passed in; otherwise
    falls back to a heuristic that maps average scaled-sensor magnitude to
    a predicted RUL (further from baseline = more degraded = lower RUL).
    """
    X = rows[FEATURE_COLS]
    if model is not None:
        return np.clip(model.predict(X.values), 0, RUL_CAP).round(1)
    magnitude = X.abs().mean(axis=1)
    return np.clip(RUL_CAP * np.exp(-magnitude), 0, RUL_CAP).round(1).to_numpy()
