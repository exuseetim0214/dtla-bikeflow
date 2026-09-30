# Concept drift and retraining

## What counts as drift

Stable weekday/weekend differences are seasonality, not automatically drift. Drift candidates include
new station launches, capacity or service-area changes, persistent station preference changes, and
model residuals that degrade after labels arrive.

## Signals

- rolling seven-day MAE divided by reference MAE;
- rolling residual mean and variance;
- PSI for demand lags, station mix, and inventory ratio;
- new or disappearing station share;
- availability probability calibration.

`configs/drift.yaml` defaults to an MAE ratio of 1.25 or PSI of 0.20. These are experiment thresholds,
not universal constants; final values should be selected on the rolling backtest.

## Strategy comparison

Evaluate four policies at identical weekly forecast origins:

1. fixed: fit once, never retrain;
2. weekly: fit every Monday using a trailing window;
3. monthly: fit on the first forecast origin of each month;
4. drift-triggered: fit when a performance or distribution trigger fires.

For each policy record forecast metrics, number of fits, training time, trigger reasons, and days since
last fit. Compare accuracy against operational cost; the most frequently trained model is not
automatically the best policy.

The test quarter remains untouched until the strategy and thresholds are frozen using train and
validation periods.

