from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def latest_status_file(root: Path) -> Path | None:
    files = sorted(root.glob("*/*.parquet"))
    return files[-1] if files else None


def load_dashboard_data(root: Path) -> tuple[pd.DataFrame, dict]:
    artifact_root = root / "artifacts"
    prediction_path = artifact_root / "test_predictions.parquet"
    if not prediction_path.exists():
        return pd.DataFrame(), {}
    predictions = pd.read_parquet(prediction_path)
    predictions["forecast_time"] = pd.to_datetime(predictions["forecast_time"])
    current = (
        predictions.sort_values("forecast_time")
        .groupby("station_id", as_index=False)
        .tail(1)
        .copy()
    )
    station_path = root / "data" / "processed" / "stations.parquet"
    if station_path.exists():
        stations = pd.read_parquet(station_path)
        current = current.merge(
            stations[["station_id", "station_name", "status"]], on="station_id", how="left"
        )
    status_path = latest_status_file(root / "data" / "status")
    if status_path:
        status = pd.read_parquet(status_path)
        columns = [
            "station_id",
            "bikes_available",
            "docks_available",
            "total_docks",
            "station_status",
        ]
        current = current.merge(status[columns], on="station_id", how="left")
    summary_path = artifact_root / "training_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return current, summary


def run() -> None:
    import plotly.express as px
    import streamlit as st

    root = Path(__file__).resolve().parents[2]
    st.set_page_config(page_title="DTLA BikeFlow", layout="wide")
    st.title("DTLA BikeFlow")
    st.caption("Metro Bike station demand and availability monitoring")
    frame, summary = load_dashboard_data(root)
    if frame.empty:
        st.info("No predictions found. Run `bikeflow prepare` and `bikeflow train` first.")
        return

    metrics = summary.get("metrics", {}).get("model", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest forecast", str(frame["forecast_time"].max()))
    col2.metric("Stations", f"{len(frame):,}")
    col3.metric("Test MAE", f"{metrics.get('mae', float('nan')):.3f}")
    col4.metric("Test WAPE", f"{metrics.get('wape', float('nan')):.1%}")

    frame["demand_band"] = pd.qcut(
        frame["prediction"].rank(method="first"),
        q=min(4, len(frame)),
        labels=["Low", "Moderate", "High", "Very high"][: min(4, len(frame))],
    ).astype(str)
    hover = ["station_id", "prediction", "target_departures_30m"]
    for column in ("station_name", "bikes_available", "docks_available", "total_docks"):
        if column in frame:
            hover.append(column)
    figure = px.scatter_map(
        frame,
        lat="latitude",
        lon="longitude",
        color="demand_band",
        size=np.maximum(frame["prediction"], 0.15),
        hover_data=hover,
        zoom=12,
        height=620,
        color_discrete_map={
            "Low": "#2A9D8F",
            "Moderate": "#E9C46A",
            "High": "#F4A261",
            "Very high": "#E76F51",
        },
    )
    st.plotly_chart(figure, width="stretch")
    st.subheader("Dispatch review")
    display = [
        column
        for column in (
            "station_id",
            "station_name",
            "prediction",
            "bikes_available",
            "docks_available",
            "total_docks",
        )
        if column in frame
    ]
    st.dataframe(frame.sort_values("prediction", ascending=False)[display], width="stretch")


if __name__ == "__main__":
    run()
