from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bikeflow.config import load_yaml, resolve_path
from bikeflow.drift import population_stability_index, should_retrain
from bikeflow.metrics import regression_metrics
from bikeflow.modeling import _build_model, _load_panel, feature_columns


def retrain_due(
    strategy: str,
    current: pd.Timestamp,
    last_fit: pd.Timestamp | None,
    *,
    drift_pending: bool = False,
) -> bool:
    if last_fit is None:
        return True
    if strategy == "fixed":
        return False
    if strategy == "weekly":
        return True
    if strategy == "monthly":
        return current.to_period("M") != last_fit.to_period("M")
    if strategy == "drift_triggered":
        return drift_pending
    raise ValueError(f"Unknown retraining strategy: {strategy}")


def rolling_retraining_backtest(
    pipeline_config_path: str | Path = "configs/pipeline.yaml",
    model_config_path: str | Path = "configs/model.yaml",
    drift_config_path: str | Path = "configs/drift.yaml",
    *,
    strategies: list[str] | None = None,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
    training_window_days: int = 365,
    max_rows: int | None = None,
    max_training_rows: int | None = 750_000,
) -> dict[str, Any]:
    pipeline = load_yaml(pipeline_config_path)
    model_config = load_yaml(model_config_path)
    drift_config = load_yaml(drift_config_path)
    panel_root = resolve_path(pipeline["paths"]["processed_dir"]) / "panel"
    artifact_root = resolve_path(pipeline["paths"]["artifact_dir"])
    frame = _load_panel(panel_root, max_rows=max_rows)
    frame["forecast_time"] = pd.to_datetime(frame["forecast_time"])
    target = "target_departures_30m"
    features = feature_columns(frame)
    frame = frame.dropna(subset=[target, *features]).sort_values("forecast_time")

    start = pd.Timestamp(evaluation_start or pipeline["split"]["validation_start"])
    end = pd.Timestamp(evaluation_end or pipeline["split"]["test_end"])
    origins = pd.date_range(start, end, freq="7D")
    selected_strategies = strategies or drift_config["retraining_strategies"]
    results: dict[str, Any] = {}

    for strategy in selected_strategies:
        model = None
        backend = None
        last_fit: pd.Timestamp | None = None
        drift_pending = False
        reference_mae: float | None = None
        reference_feature: np.ndarray | None = None
        actual_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        periods: list[dict[str, Any]] = []
        fit_count = 0

        for origin in origins:
            period_end = min(origin + pd.Timedelta(days=7), end + pd.Timedelta(seconds=1))
            evaluation = frame.loc[
                frame["forecast_time"].ge(origin) & frame["forecast_time"].lt(period_end)
            ]
            if evaluation.empty:
                continue
            did_fit = retrain_due(strategy, origin, last_fit, drift_pending=drift_pending)
            if did_fit:
                training_start = origin - pd.Timedelta(days=training_window_days)
                training = frame.loc[
                    frame["forecast_time"].lt(origin) & frame["forecast_time"].ge(training_start)
                ]
                if max_training_rows and len(training) > max_training_rows:
                    training = training.tail(max_training_rows)
                if training.empty:
                    continue
                model, backend = _build_model(model_config)
                if backend == "lightgbm":
                    categorical = ["station_id", "hour", "day_of_week", "month"]
                    for column in categorical:
                        training[column] = training[column].astype("category")
                        evaluation[column] = evaluation[column].astype("category")
                    model.fit(
                        training[features],
                        training[target],
                        categorical_feature=categorical,
                    )
                else:
                    model.fit(training[features], training[target])
                last_fit = origin
                fit_count += 1
                if reference_feature is None:
                    reference_feature = training["departures_roll_sum_336"].to_numpy()
            if model is None:
                continue

            prediction = np.clip(model.predict(evaluation[features]), 0, None)
            period_metrics = regression_metrics(evaluation[target], prediction)
            if reference_mae is None:
                reference_mae = period_metrics["mae"]
            psi = population_stability_index(
                reference_feature,
                evaluation["departures_roll_sum_336"].to_numpy(),
            )
            decision = should_retrain(
                reference_mae=reference_mae,
                recent_mae=period_metrics["mae"],
                feature_psi=psi,
                mae_ratio_threshold=drift_config["triggers"]["rolling_mae_ratio"],
                psi_threshold=drift_config["triggers"]["feature_psi"],
            )
            drift_pending = bool(decision["trigger"])
            actual_parts.append(evaluation[target].to_numpy())
            prediction_parts.append(prediction)
            periods.append(
                {
                    "start": origin.isoformat(),
                    "end": period_end.isoformat(),
                    "rows": len(evaluation),
                    "fit": did_fit,
                    "metrics": period_metrics,
                    "feature_psi": psi,
                    "drift": decision,
                }
            )

        if actual_parts:
            overall = regression_metrics(
                np.concatenate(actual_parts), np.concatenate(prediction_parts)
            )
        else:
            overall = {}
        results[strategy] = {
            "backend": backend,
            "fit_count": fit_count,
            "metrics": overall,
            "periods": periods,
        }

    summary = {
        "evaluation_start": start.isoformat(),
        "evaluation_end": end.isoformat(),
        "training_window_days": training_window_days,
        "strategies": results,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path = artifact_root / "retraining_backtest.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["output_path"] = str(output_path)
    return summary
