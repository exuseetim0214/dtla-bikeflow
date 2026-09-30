# Evaluation protocol

## Fixed holdout

| Split | Dates | Purpose |
|---|---|---|
| Train | 2024-04-01–2025-12-31 | Fit parameters and historical encodings |
| Validation | 2026-01-01–2026-03-31 | Tune model and early stopping |
| Test | 2026-04-01–2026-06-30 | One untouched final comparison |

Random row splits are prohibited. Rows for one forecast origin must stay in the same split across all
stations.

## Demand metrics

- MAE: primary station-window error;
- RMSE: emphasizes large operational misses;
- WAPE: network-level absolute error normalized by observed trips;
- Bias: detects systematic over/under-forecasting.

MAPE is not a primary metric because the panel contains many zero targets.

## Required slices

Report metrics by 30/60-minute horizon, station traffic quartile, hour, weekday/weekend, month, known
versus new station, and zero versus positive target. Also report network totals to ensure a low average
does not hide poor high-demand performance.

## Deployment gate

The main model must beat the seasonal mean on overall MAE and on the top traffic quartile, remain
within an agreed bias tolerance, and produce non-negative predictions. Availability probabilities
must additionally be calibrated and improve Recall@K over ranking by current inventory alone.

