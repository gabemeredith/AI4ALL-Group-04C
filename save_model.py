"""
Train and save the Random Forest RUL model for the Streamlit dashboard.

Trains on nasa_cmapss_FD001_scaled.csv (the committed, labeled FD001 dataset).
Evaluates on an engine-grouped hold-out split so no engine leaks across
train/test, then refits on all 100 engines and saves the deployable model.

Usage:  python save_model.py
Output: random_forest_model.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from model import FEATURE_COLS, MODEL_PATH, RUL_CAP

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"
ID_COL = "unit_number"


def cmapss_score(y_true, y_pred):
    """Asymmetric PHM score (lower is better). Late predictions — engine
    predicted to last LONGER than it does — are the dangerous ones and are
    penalized harder (/10) than early predictions (/13)."""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def build_model():
    # Tuned on a grouped hold-out: shallower leaves + sqrt feature sampling
    # generalize better than the default here (RMSE 18.32 -> 18.03).
    return RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def main():
    df = pd.read_csv(DATA_PATH)

    # Piecewise-linear RUL: cap so the model isn't forced to fit long, flat,
    # unpredictable early-life stretches. The CSV ships raw (uncapped) RUL.
    X = df[FEATURE_COLS].values
    y = np.minimum(df["RUL"].values, RUL_CAP)
    groups = df[ID_COL].values

    # Engine-grouped hold-out: test engines are entirely unseen in training.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))

    print(f"Evaluating on a grouped hold-out "
          f"({np.unique(groups[tr]).size} train / "
          f"{np.unique(groups[te]).size} test engines)...")
    model = build_model()
    model.fit(X[tr], y[tr])
    pred = np.clip(model.predict(X[te]), 0, RUL_CAP)

    rmse = np.sqrt(mean_squared_error(y[te], pred))
    mae = mean_absolute_error(y[te], pred)
    r2 = r2_score(y[te], pred)
    score = cmapss_score(y[te], pred)

    print("\n=== Random Forest — held-out performance ===")
    print(f"RMSE          : {rmse:.2f} cycles")
    print(f"MAE           : {mae:.2f} cycles")
    print(f"R^2           : {r2:.3f}")
    print(f"C-MAPSS score : {score:.1f}  (lower is better)")

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )
    print("\nTop features:")
    for name, imp in importances[:6]:
        print(f"  {name:22s} {imp:.3f}")

    # Refit on ALL engines for the deployed model — the dashboard should use
    # every row we have, not just the 75% training split.
    print("\nRefitting on all 100 engines and saving...")
    final = build_model()
    final.fit(X, y)
    joblib.dump(final, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
