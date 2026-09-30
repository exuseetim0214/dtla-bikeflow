# End-to-end smoke training

This run validates the complete local path from prepared Parquet partitions through chronological
splitting, model fitting, holdout prediction, baseline comparison, metrics, and artifact export.

## Scope

- Invocation: `bikeflow train --max-rows 500000`
- Rows after feature warm-up: 378,610 train / 51,840 validation / 52,079 test
- Stations: a deterministic subset selected only for the smoke run
- Backend: scikit-learn histogram gradient boosting with Poisson loss
- Reason for fallback: the local macOS runtime did not provide `libomp` for LightGBM

## Result

| Model | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Poisson gradient boosting | 0.2392 | 0.5023 | 1.1426 | -0.0098 |
| Seasonal mean baseline | 0.2542 | 0.5703 | 1.2145 | -0.0058 |

The learned model beat the seasonal baseline on this deterministic subset. These values are a
pipeline smoke test, not the final project result. The final report must use the full station panel,
the frozen configuration, and LightGBM or an explicitly selected alternative.

