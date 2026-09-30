# Demand modeling

## Primary question

At every completed 30-minute window, estimate the number of departures and arrivals for each known
DTLA station during the next 30 and 60 minutes.

## Baselines

1. Last-week same slot (`lag_336`).
2. Mean of the previous 1–4 weekly slots (`lags 336, 672, 1008, 1344`).
3. Recent rolling flow for diagnostic comparison.

The seasonal mean is the minimum bar for a useful model. A learned model that does not beat it on the
chronological holdout should not be deployed.

## Main estimator

The default LightGBM regressor uses a Poisson objective because the targets are non-negative counts
with many zeros. The fallback estimator is scikit-learn histogram gradient boosting with Poisson loss.
Predictions are clipped at zero.

Features include station identity and coordinates, station age, calendar cycles, demand and arrival
lags, rolling flows, and net flow. Passenger plan and bike type are not read from the future target
window. If used later, they must be historical rolling shares.

## Spatial information

`bikeflow.spatial.nearest_station_map` computes Haversine neighbors. A production spatial feature can
aggregate lagged demand across the nearest five stations or within one kilometer. It must join only
lagged values at the same forecast origin; using neighbors' target-window demand is leakage.

## Cold start

For a new station, station-specific historical lags are unavailable. The fallback feature set is:

- latitude and longitude;
- neighbor lagged demand and net flow;
- hour/day/month seasonality;
- station age;
- network-wide demand level.

Cold-start results should be reported as a separate evaluation slice.

