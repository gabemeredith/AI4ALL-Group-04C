"""
Streamlit dashboard for the AI4ALL Group 04C turbofan RUL project.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"

st.set_page_config(page_title="Turbofan RUL Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


df = load_data()

with st.sidebar:
    st.title("Turbofan Engine RUL Dashboard")
    st.markdown(
        "**AI4ALL Group 04C** — predicting a turbofan engine's Remaining "
        "Useful Life (RUL) from NASA C-MAPSS sensor telemetry, to support "
        "predictive maintenance instead of fixed replacement schedules."
    )
    st.divider()
    engine_ids = sorted(df["unit_number"].unique().tolist())
    selected_engine = st.selectbox("Select engine", engine_ids)

engine_df = df[df["unit_number"] == selected_engine].sort_values("time_in_cycles")
min_cycle = int(engine_df["time_in_cycles"].min())
max_cycle = int(engine_df["time_in_cycles"].max())

st.header(f"Engine {selected_engine}")
st.caption(f"{len(engine_df)} recorded cycles ({min_cycle}-{max_cycle}).")
