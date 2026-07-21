"""
Run this once to train and save the Random Forest model as a .pkl file
for use with the Streamlit app.

Usage: python save_model.py
Output: random_forest_model.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA_PATH = "train_FD001_scaled.csv"
DROP_COLS = ["op_setting_3", "sensor_1", "sensor_5", "sensor_6",
             "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
df = df.join(max_cycle, on="unit")
df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=125)
df = df.drop(columns=["max_cycle"])

feature_cols = [c for c in df.columns if c not in ["unit", "cycle", "RUL"]]
X = df[feature_cols].values
y = df["RUL"].values

print("Training Random Forest...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X, y)

joblib.dump(model, "random_forest_model.pkl")
print("Saved: random_forest_model.pkl")
