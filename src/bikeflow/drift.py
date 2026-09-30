from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: object,
    current: object,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("PSI requires finite reference and current observations")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(reference_values, bins=edges)[0] / reference_values.size
    cur_hist = np.histogram(current_values, bins=edges)[0] / current_values.size
    ref_hist = np.clip(ref_hist, epsilon, None)
    cur_hist = np.clip(cur_hist, epsilon, None)
    return float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))


def should_retrain(
    *,
    reference_mae: float,
    recent_mae: float,
    feature_psi: float,
    mae_ratio_threshold: float = 1.25,
    psi_threshold: float = 0.20,
) -> dict[str, object]:
    mae_ratio = recent_mae / reference_mae if reference_mae > 0 else float("inf")
    reasons = []
    if mae_ratio >= mae_ratio_threshold:
        reasons.append("rolling_mae")
    if feature_psi >= psi_threshold:
        reasons.append("feature_psi")
    return {"trigger": bool(reasons), "reasons": reasons, "mae_ratio": mae_ratio}
