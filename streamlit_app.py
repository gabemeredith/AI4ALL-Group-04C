"""
RUL Prediction App — Team 04C
AI4ALL Ignite SU26 | NASA C-MAPSS Turbofan

Run with: streamlit run streamlit_app.py
Requires: streamlit, numpy, pandas, scikit-learn, joblib
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Turbofan Engine RUL Predictor")
st.caption("Team 04C · AI4ALL Ignite SU26 · NASA C-MAPSS FD001")
st.markdown("---")

# ── Feature list (matches training) ───────────────────────────────────────────
FEATURE_COLS = [
    "op_setting_1", "op_setting_2",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "random_forest_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# ── Sidebar: input mode ───────────────────────────────────────────────────────
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio("Choose input method:", ["Upload CSV", "Manual Input"])

# ── CSV Upload Mode ───────────────────────────────────────────────────────────
if input_mode == "Upload CSV":
    st.subheader("Upload Sensor Data")
    st.markdown("Upload a pre-scaled CSV (z-score standardized, same format as Tom's FD001).")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = df[FEATURE_COLS].values
            if model:
                preds = np.clip(model.predict(X), 0, 125)
                df["Predicted RUL"] = preds.round(1)
                st.success("Predictions complete!")
                st.dataframe(df[["unit", "cycle", "Predicted RUL"]] if "unit" in df.columns else df[FEATURE_COLS + ["Predicted RUL"]])
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions CSV", csv_out, "rul_predictions.csv", "text/csv")
            else:
                st.warning("No trained model found. Place `random_forest_model.pkl` in the same folder.")

# ── Manual Input Mode ─────────────────────────────────────────────────────────
else:
    st.subheader("Manual Sensor Input")
    st.markdown("Enter a single engine's sensor readings (z-score standardized).")

    col1, col2 = st.columns(2)
    inputs = {}
    for i, feat in enumerate(FEATURE_COLS):
        col = col1 if i % 2 == 0 else col2
        inputs[feat] = col.number_input(feat, value=0.0, format="%.4f")

    if st.button("Predict RUL", type="primary"):
        X = np.array([[inputs[f] for f in FEATURE_COLS]])
        if model:
            pred = np.clip(model.predict(X)[0], 0, 125)
            st.metric("Predicted Remaining Useful Life", f"{pred:.1f} cycles")
            if pred < 30:
                st.error("⚠️ Engine approaching end of life — maintenance recommended soon.")
            elif pred < 60:
                st.warning("🔶 Moderate degradation detected — schedule inspection.")
            else:
                st.success("✅ Engine health looks good.")
        else:
            st.warning("No trained model found. Place `random_forest_model.pkl` in the same folder.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Random Forest · RMSE 20.11 · MAE 15.09 on FD001 · RUL capped at 125 cycles")
