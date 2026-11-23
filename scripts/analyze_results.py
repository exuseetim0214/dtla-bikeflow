"""Audit saved forecasts and calculate descriptive operating patterns without retraining.

Run from the repository root: python scripts/analyze_results.py
The generated JSON is a reproducible evidence source for reports/full_results_report.md.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from bikeflow.metrics import regression_metrics
from bikeflow.prepare import load_station_table

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_departures_30m"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(frame: pd.DataFrame) -> dict:
    actual = frame[TARGET].to_numpy(dtype=float)
    model = frame["prediction"].to_numpy(dtype=float)
    baseline = frame["seasonal_baseline"].to_numpy(dtype=float)
    return {
        "rows": len(frame),
        "stations": int(frame["station_id"].nunique()),
        "observed_trips": float(actual.sum()),
        "zero_fraction": float((actual == 0).mean()),
        "mean_target": float(actual.mean()),
        "model": regression_metrics(actual, model),
        "seasonal_baseline": regression_metrics(actual, baseline),
        "zero_baseline": regression_metrics(actual, np.zeros(len(actual))),
        "predicted_total": float(model.sum()),
        "seasonal_total": float(baseline.sum()),
    }


def station_performance(frame: pd.DataFrame, station_names: pd.Series) -> list[dict]:
    records = []
    for station_id, group in frame.groupby("station_id"):
        result = comparison(group)
        result.update(
            station_id=int(station_id),
            station_name=str(station_names.get(station_id, station_id)),
        )
        result["mae_improvement_percent"] = 100 * (
            1 - result["model"]["mae"] / result["seasonal_baseline"]["mae"]
        )
        result["rmse_improvement_percent"] = 100 * (
            1 - result["model"]["rmse"] / result["seasonal_baseline"]["rmse"]
        )
        records.append(result)
    return sorted(records, key=lambda row: row["observed_trips"], reverse=True)


def training_volume_groups(prediction_station_ids: list[int]) -> dict[int, str]:
    """Assign station volume quartiles using training labels, never test volume."""
    volumes = {}
    for station_id in prediction_station_ids:
        path = ROOT / "data/processed/panel" / f"station_id={station_id}" / "panel.parquet"
        frame = pd.read_parquet(path, columns=["forecast_time", TARGET])
        time = pd.to_datetime(frame["forecast_time"])
        history = frame.loc[time.ge("2024-04-01") & time.lt("2026-01-01"), TARGET]
        volumes[station_id] = float(history.sum()) if history.notna().any() else np.nan
    values = pd.Series(volumes).dropna()
    quartiles = pd.qcut(
        values.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    )
    result = quartiles.astype(str).to_dict()
    for station_id in prediction_station_ids:
        result.setdefault(station_id, "No training history")
    return result


def paired_week_bootstrap(frame: pd.DataFrame, draws: int = 2000) -> dict:
    """Exploratory calendar-week block bootstrap, not an independence guarantee."""
    time = pd.to_datetime(frame["forecast_time"])
    weekly = (
        pd.DataFrame(
            {
                "week": time.dt.to_period("W-SUN").astype(str),
                "absolute_error_difference": (
                    (frame["prediction"] - frame[TARGET]).abs()
                    - (frame["seasonal_baseline"] - frame[TARGET]).abs()
                ),
                "n": 1,
            }
        )
        .groupby("week")
        .agg({"absolute_error_difference": "sum", "n": "sum"})
    )
    rng = np.random.default_rng(42)
    sample = rng.integers(0, len(weekly), size=(draws, len(weekly)))
    differences = weekly["absolute_error_difference"].to_numpy()[sample].sum(axis=1)
    counts = weekly["n"].to_numpy()[sample].sum(axis=1)
    lower, upper = np.quantile(differences / counts, [0.025, 0.975])
    return {
        "method": "Paired calendar-week block bootstrap; exploratory",
        "draws": draws,
        "seed": 42,
        "calendar_week_blocks": len(weekly),
        "mae_difference_model_minus_seasonal": float(
            weekly["absolute_error_difference"].sum() / weekly["n"].sum()
        ),
        "interval_95_percent": [float(lower), float(upper)],
        "caveat": (
            "One quarter and few week blocks; does not prove stationarity, causal benefit, "
            "or operational effectiveness."
        ),
    }


def descriptive_station_flows(stations: pd.DataFrame) -> dict:
    """Use raw trips, independently of the saved forecast subset."""
    candidates = stations.loc[
        stations["region"].eq("DTLA")
        & stations["status"].eq("Active")
        & stations["go_live_date"].le(pd.Timestamp("2025-04-01"))
    ]
    station_ids = set(candidates["station_id"].astype(int))
    names = candidates.set_index("station_id")["station_name"]
    windows = [("07:00-10:00", 7, 10), ("16:00-19:00", 16, 19)]
    records = []
    quarterly_activity = []
    for year in (2024, 2025, 2026):
        path = ROOT / f"data/raw/trips/metro-trips-{year}-q2.csv"
        trips = pd.read_csv(
            path, usecols=["start_time", "end_time", "start_station", "end_station"]
        )
        event_parts = []
        for side, kind in (("start", "departures"), ("end", "arrivals")):
            events = trips.loc[
                trips[f"{side}_station"].isin(station_ids),
                [f"{side}_station", f"{side}_time"],
            ].copy()
            events.columns = ["station_id", "time"]
            events["time"] = pd.to_datetime(events["time"], errors="coerce")
            events = events.loc[
                events["time"].ge(f"{year}-04-01") & events["time"].lt(f"{year}-07-01")
            ]
            events["kind"] = kind
            event_parts.append(events)
        events = pd.concat(event_parts, ignore_index=True)
        events["date"] = events["time"].dt.normalize()
        events["hour"] = events["time"].dt.hour
        dates = pd.date_range(f"{year}-04-01", f"{year}-06-30")
        for weekend in (False, True):
            eligible_days = dates[(dates.dayofweek >= 5) == weekend]
            subset = events.loc[events["date"].isin(eligible_days)]
            for label, first_hour, last_hour in windows:
                selected = subset.loc[subset["hour"].ge(first_hour) & subset["hour"].lt(last_hour)]
                daily = selected.groupby(["station_id", "date", "kind"]).size()
                full_index = pd.MultiIndex.from_product(
                    [sorted(station_ids), eligible_days, ["departures", "arrivals"]],
                    names=["station_id", "date", "kind"],
                )
                daily = daily.reindex(full_index, fill_value=0).unstack("kind")
                daily["loss"] = daily["departures"] - daily["arrivals"]
                for station_id, group in daily.groupby(level="station_id"):
                    records.append(
                        {
                            "year": year,
                            "station_id": int(station_id),
                            "station_name": str(names.loc[station_id]),
                            "day_type": "Weekend" if weekend else "Weekday",
                            "window": label,
                            "days": len(eligible_days),
                            "departures_per_day": float(group["departures"].mean()),
                            "arrivals_per_day": float(group["arrivals"].mean()),
                            "net_loss_per_day": float(group["loss"].mean()),
                            "net_loss_day_fraction": float(group["loss"].gt(0).mean()),
                            "net_loss_p75": float(group["loss"].quantile(0.75)),
                        }
                    )
        quarterly_activity.append({"year": year, "network_trips": len(trips)})
    return {
        "selection": (
            "Official DTLA region, Active in station table dated 2026-07-15, "
            "go-live on or before 2025-04-01. Weekdays include public holidays. "
            "Each quarter is a trip-start cohort; arrivals from pre-quarter starts are unavailable."
        ),
        "quarterly_network_activity": quarterly_activity,
        "station_windows": records,
    }


def main() -> None:
    artifact = ROOT / "artifacts"
    predictions_path = artifact / "test_predictions.parquet"
    summary = json.loads((artifact / "training_summary.json").read_text())
    frame = pd.read_parquet(predictions_path)
    frame["forecast_time"] = pd.to_datetime(frame["forecast_time"])
    assert not frame.duplicated(["station_id", "forecast_time"]).any()
    assert np.isfinite(frame[[TARGET, "prediction", "seasonal_baseline"]]).all().all()
    assert frame[[TARGET, "prediction", "seasonal_baseline"]].ge(0).all().all()
    assert len(frame) == summary["rows"]["test"]

    stations = load_station_table(ROOT / "data/raw")
    names = stations.set_index("station_id")["station_name"]
    overall = comparison(frame)
    for model_key, summary_key in (("model", "model"), ("seasonal_baseline", "seasonal_baseline")):
        for metric in ("mae", "rmse", "wape", "bias"):
            assert np.isclose(
                overall[model_key][metric],
                summary["metrics"][summary_key][metric],
                rtol=1e-10,
                atol=1e-12,
            )

    groups = training_volume_groups(sorted(frame["station_id"].unique().tolist()))
    frame["training_volume_quartile"] = frame["station_id"].map(groups)
    frame["hour"] = frame["forecast_time"].dt.hour
    frame["day_type"] = np.where(frame["forecast_time"].dt.dayofweek < 5, "Weekday", "Weekend")
    frame["month"] = frame["forecast_time"].dt.strftime("%Y-%m")
    frame["target_type"] = np.where(frame[TARGET].eq(0), "Zero departures", "Positive departures")

    slices = {}
    for field in ("day_type", "month", "hour", "training_volume_quartile", "target_type"):
        slices[field] = {str(key): comparison(group) for key, group in frame.groupby(field)}
        assert sum(value["rows"] for value in slices[field].values()) == len(frame)

    daily = frame.groupby(frame["forecast_time"].dt.normalize())[
        [TARGET, "prediction", "seasonal_baseline"]
    ].sum()
    daily["station_id"] = 0
    daily_comparison = comparison(daily)
    daily_rows = [
        {
            "date": date.isoformat(),
            "actual": float(row[TARGET]),
            "model": float(row["prediction"]),
            "seasonal": float(row["seasonal_baseline"]),
        }
        for date, row in daily.iterrows()
    ]
    bundle = joblib.load(artifact / "demand_model.joblib")
    model = bundle["model"]
    trained_at = pd.Timestamp(bundle["trained_at_utc"])
    prepared = json.loads((artifact / "prepare_summary.json").read_text())
    panel_station_ids = {item["station_id"] for item in prepared["partitions"]}
    prediction_ids = set(frame["station_id"].astype(int))
    performance = station_performance(frame, names)
    result = {
        "provenance": {
            "trained_at_utc": trained_at.isoformat(),
            "trained_at_los_angeles": trained_at.tz_convert(
                ZoneInfo("America/Los_Angeles")
            ).isoformat(),
            "backend": bundle["backend"],
            "target": bundle["target"],
            "feature_count": len(bundle["features"]),
            "feature_names": bundle["features"],
            "actual_estimator_parameters": model.get_params(),
            "actual_iterations": int(model.n_iter_),
            "internal_early_stopping_enabled": bool(model.do_early_stopping_),
            "installed_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow")
            },
            "input_sha256": {
                str(path.relative_to(ROOT)): file_hash(path)
                for path in (
                    artifact / "training_summary.json",
                    predictions_path,
                    artifact / "demand_model.joblib",
                    ROOT / "configs/pipeline.yaml",
                    ROOT / "configs/model.yaml",
                    ROOT / "configs/data_manifest.json",
                )
            },
            "config_snapshot_warning": (
                "Configuration hashes describe current files. Training did not save frozen "
                "configuration snapshots, so historical file identity cannot be proved."
            ),
        },
        "split_rows": summary["rows"],
        "coverage": {
            "prepared_stations": prepared["stations_written"],
            "prepared_rows": prepared["panel_rows"],
            "test_stations": len(prediction_ids),
            "prepared_stations_absent_from_test": sorted(panel_station_ids - prediction_ids),
            "test_start": frame["forecast_time"].min().isoformat(),
            "test_end": frame["forecast_time"].max().isoformat(),
            "test_days": len(daily),
        },
        "overall": overall,
        "relative_improvement_over_seasonal_percent": {
            metric: 100 * (1 - overall["model"][metric] / overall["seasonal_baseline"][metric])
            for metric in ("mae", "rmse", "wape")
        },
        "slices": slices,
        "station_performance": performance,
        "station_win_counts": {
            "total": len(performance),
            "mae_better_than_seasonal": sum(
                row["model"]["mae"] < row["seasonal_baseline"]["mae"] for row in performance
            ),
            "rmse_better_than_seasonal": sum(
                row["model"]["rmse"] < row["seasonal_baseline"]["rmse"] for row in performance
            ),
        },
        "daily_network": daily_comparison,
        "daily_network_series": daily_rows,
        "paired_mae_uncertainty": paired_week_bootstrap(frame),
        "operational_patterns": descriptive_station_flows(stations),
        "checks": {
            "saved_metrics_reproduced": True,
            "predictions_finite_nonnegative": True,
            "unique_station_forecast_keys": True,
            "slice_rows_reconcile": True,
        },
    }

    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    path = ROOT / "reports/full_run_analysis.json"
    path.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis": str(path),
                "coverage": result["coverage"],
                "overall": overall,
                "station_win_counts": result["station_win_counts"],
                "paired_mae_uncertainty": result["paired_mae_uncertainty"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
