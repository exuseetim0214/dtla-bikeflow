# DTLA BikeFlow

DTLA BikeFlow is a reproducible station-level forecasting project for Metro Bike Share. It converts
nine official quarterly trip files into a station-by-30-minute panel, predicts departures, arrivals,
and net flow at 30/60-minute horizons, and provides the collection and labeling path required for
real empty/full station risk.

For actual implementation details and the full run completed on November 22, 2026, start with the
[step-by-step guide](docs/step_by_step_guide.md) and
[results report](reports/full_results_report.md). The saved model forecasts 30-minute departures;
arrival/60-minute targets are prepared but those estimators and availability classifiers are not
trained. Across 150 test stations, MAE is 0.140041 and RMSE 0.338774. The seasonal-baseline MAE gain is
only 0.17%, and the always-zero predictor has lower MAE, so the current model is experimental.

## Included

- Official-data downloader with atomic writes and SHA-256 manifests.
- Nine quarters from 2024 Q2 through 2026 Q2: 1,027,489 trips.
- DTLA station filtering through the official Station Table.
- Cross-boundary arrival and departure accounting.
- Leakage-safe lag, rolling, calendar, and spatial features.
- Seasonal baselines and Poisson LightGBM with a scikit-learn fallback.
- Chronological train/validation/test protocol.
- Live GeoJSON status collector and persistent empty/full label construction.
- PSI and rolling-error drift primitives.
- Streamlit/Plotly map dashboard, tests, CI, and detailed design documentation.

## Quick start

```bash
bash scripts/setup.sh
source .venv/bin/activate
bikeflow download
bikeflow audit
bikeflow prepare
bikeflow train
bash scripts/dashboard.sh
```

Run `bikeflow collect-status` every five minutes to begin building the historical inventory data
needed for availability-risk classification. Trip logs alone cannot reconstruct operator rebalancing.

See the [Chinese README](README.md) and [documentation index](docs/README.md) for the full design,
metrics, limitations, and operational workflow.

