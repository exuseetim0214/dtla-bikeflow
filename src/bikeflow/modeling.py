from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from bikeflow.baselines import seasonal_mean_prediction
from bikeflow.config import load_yaml, resolve_path
from bikeflow.metrics import regression_metrics
from bikeflow.splits import assign_time_split

NON_FEATURE_COLUMNS = {
    "forecast_time",
    "target_departures_30m",
    "target_departures_60m",
    "target_arrivals_30m",
    "target_arrivals_60m",
    "target_net_flow_30m",
    "target_net_flow_60m",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS and not column.startswith("target_")
    ]


def _load_panel(panel_root: Path, max_rows: int | None = None) -> pd.DataFrame:
    files = sorted(panel_root.glob("station_id=*/panel.parquet"))
    if not files:
        raise FileNotFoundError("Prepared panel missing. Run `bikeflow prepare` first.")
    frames: list[pd.DataFrame] = []
    remaining = max_rows
    for path in files:
        frame = pd.read_parquet(path)
        if remaining is not None:
            if remaining <= 0:
                break
            frame = frame.head(remaining)
            remaining -= len(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _build_model(config: dict[str, Any]) -> tuple[Any, str]:
    params = dict(config["regression"]["params"])
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(objective="poisson", **params), "lightgbm"
    except (ImportError, OSError) as error:
        warnings.warn(
            f"LightGBM is unavailable ({error}); using HistGradientBoostingRegressor.",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("n_estimators", 500),
            max_leaf_nodes=params.get("num_leaves", 63),
            l2_regularization=params.get("reg_lambda", 1.0),
            random_state=params.get("random_state", 42),
        )
        return fallback, "hist_gradient_boosting"


def train_demand_model(
    pipeline_config_path: str | Path = "configs/pipeline.yaml",
    model_config_path: str | Path = "configs/model.yaml",
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    pipeline_config = load_yaml(pipeline_config_path)
    model_config = load_yaml(model_config_path)
    panel_root = resolve_path(pipeline_config["paths"]["processed_dir"]) / "panel"
    artifact_root = resolve_path(pipeline_config["paths"]["artifact_dir"])
    frame = _load_panel(panel_root, max_rows=max_rows)
    frame["split"] = assign_time_split(frame, pipeline_config["split"])
    target = "target_departures_30m"
    features = feature_columns(frame)
    usable = frame.dropna(subset=[target, *features]).copy()
    train = usable.loc[usable["split"].eq("train")]
    validation = usable.loc[usable["split"].eq("validation")]
    test = usable.loc[usable["split"].eq("test")]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            f"Empty split after feature warm-up: train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}"
        )

    model, backend = _build_model(model_config)
    fit_kwargs: dict[str, Any] = {}
    if backend == "lightgbm":
        fit_kwargs = {
            "eval_set": [(validation[features], validation[target])],
            "callbacks": [],
            "categorical_feature": ["station_id", "hour", "day_of_week", "month"],
        }
        for column in fit_kwargs["categorical_feature"]:
            train[column] = train[column].astype("category")
            validation[column] = validation[column].astype("category")
            test[column] = test[column].astype("category")
    model.fit(train[features], train[target], **fit_kwargs)

    prediction = np.clip(model.predict(test[features]), 0, None)
    baseline = seasonal_mean_prediction(test)
    metrics = {
        "model": regression_metrics(test[target], prediction),
        "seasonal_baseline": regression_metrics(test[target], baseline),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    bundle_path = artifact_root / "demand_model.joblib"
    joblib.dump(
        {
            "model": model,
            "backend": backend,
            "features": features,
            "target": target,
            "trained_at_utc": datetime.now(UTC).isoformat(),
        },
        bundle_path,
    )
    predictions = test[["station_id", "forecast_time", target, "latitude", "longitude"]].copy()
    predictions["prediction"] = prediction
    predictions["seasonal_baseline"] = baseline.to_numpy()
    prediction_path = artifact_root / "test_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    summary = {
        "backend": backend,
        "features": features,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "metrics": metrics,
        "model_path": str(bundle_path),
        "prediction_path": str(prediction_path),
    }
    summary_path = artifact_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
