# DTLA BikeFlow: implementation and reproduction guide

This guide explains how the repository turns Metro Bike Share trip records into station-level
30-minute departure forecasts, how to reproduce the current run, and how to interpret its outputs.
For measured results from the full run completed on November 22, 2026, read the
[results report](../reports/full_results_report.md).

## 1. Define the prediction and the operational decision

The implemented question is: **At time `t`, how many recorded rentals will start at station `s`
during `[t, t + 30 minutes)`?**

For example, at 08:00, we use already observed activity to predict departures between 08:00 and
08:30 at 7th & Flower. A prediction of 1.7 is an expected trip count, not a physical fractional bike.
Do not round predictions before aggregating or evaluating them.

The longer-term decision is whether to replenish a station or preserve room for returns. That
decision additionally needs arrivals, current inventory, usable capacity, and uncertainty. The
currently trained model forecasts recorded departures, not total latent demand: riders who find an
empty station cannot generate a rental record, so unmet demand is unobserved.

The data builder creates six targets: departures, arrivals, and arrivals-minus-departures, each at
30 and 60 minutes. The current `train` command fits **only `target_departures_30m`**. There is no
trained empty/full classifier, 60-minute model, or routing optimizer in the saved run.

## 2. Open the repository and activate its environment

On the existing computer:

```bash
cd "/Users/exusee_tim/Documents/ChatGPT/bike project"
source .venv/bin/activate
bikeflow --help
```

On a fresh installation, first run from the repository root:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script creates a virtual environment and installs the Python package plus ML, application,
and development dependencies. It requires Python 3.11 or later and internet access. Never activate
one interpreter and install with a different interpreter; use `python -m pip` inside the environment.

The project is organized as follows:

| Location | Responsibility |
|---|---|
| `configs/data_manifest.json` | Exact quarterly and station-table source URLs |
| `configs/pipeline.yaml` | Paths, feature windows, station region, time splits |
| `configs/model.yaml` | Preferred estimator parameters |
| `configs/drift.yaml` | Experimental drift thresholds and retraining policies |
| `src/bikeflow/` | Download, audit, features, model, metrics, monitoring, Dashboard |
| `data/raw/` | Original source files and download hashes |
| `data/processed/` | Model-ready station partitions |
| `artifacts/` | Current model, test predictions, and summaries |
| `reports/` | Human-readable findings and reproducible analysis |
| `tests/` | Checks on features, metrics, CLI, spatial calculations, and utilities |

## 3. Acquire the source data

```bash
bikeflow download
```

The source registry specifies nine quarters, from 2024 Q2 through 2026 Q2. This is **27 months**,
slightly more than two years. The pipeline also downloads the official station table dated
2026-07-15. The source landing page is [Metro Bike Share Data](https://bikeshare.metro.net/about/data/).

For each quarterly ZIP, `download.py`:

1. Downloads into a temporary file with HTTPS certificate verification enabled.
2. Renames the completed file into the archive directory.
3. Requires one CSV archive member and checks its path.
4. Extracts that CSV into a canonical quarter filename.
5. Records byte counts and SHA-256 digests for the ZIP and CSV.

The outputs are:

```text
data/raw/archives/metro-trips-2026-q2.zip
data/raw/trips/metro-trips-2026-q2.csv
data/raw/stations/stations-2026-07-15.csv
data/raw/manifests/download_manifest.json
```

Existing files are reused; `--force` explicitly replaces them. Hashes record downloaded content, but
there is no independently published expected-hash registry in this repository. The downloader does
not automatically discover newly published quarters: update the source registry to include them.

## 4. Audit the records

```bash
bikeflow audit
```

The audit reads CSVs in chunks and checks expected columns, missing values, within-quarter duplicate
trip IDs, timestamp coverage, station matches, and DTLA flow counts. Read:

```text
reports/data_audit.md
reports/data_audit.json
```

The downloaded files contain 1,027,489 trips and 273 distinct station IDs across the network. All
observed station IDs match the downloaded station table. DTLA accounts for 593,813 starts and
576,836 ends across the nine trip-start cohorts.

Audit and cleaning are separate concepts. The audit reports duplicates and missingness; the demand
builder does not silently deduplicate every raw file or repair all anomalous records. It relies on
the published source plus valid IDs and parseable event timestamps. Trip duration, bike type, and
passholder type do not enter the current feature builder, so modifying them would not affect this run.

## 5. Define the station set

The official table is normalized by trimming column names, converting IDs and coordinates to numeric
types, parsing launch dates, and selecting `region == "DTLA"`. Stations also need coordinates and a
launch date to enter the panel. Station 3000 is a virtual operations station and is excluded as an
entity to forecast.

This uses Metro's region label rather than a custom geographic polygon. Verify that the resulting
service-area definition matches a strict Downtown LA boundary if that is needed for the project.
The builder does not limit historical training rows to stations currently marked Active, because
retired stations may have valid historical activity.

There are 180 observed DTLA station partitions in the prepared dataset. The full saved test set
contains 150 stations. A historical station can be absent from the test set because it no longer
appears during the test period or because it lacks sufficient lag history.

## 6. Aggregate departures and arrivals independently

The builder reads these four raw columns:

```text
start_station, start_time, end_station, end_time
```

For departures, it groups events by `start_station` and the 30-minute bucket of `start_time`.
For arrivals, it groups by `end_station` and the bucket of `end_time`.

Filtering is applied independently to each endpoint. A trip from outside DTLA into DTLA contributes
an arrival, and a trip from DTLA to outside contributes a departure. Filtering first to trips with
both endpoints in DTLA would systematically lose those movements.

The current timestamps are parsed as local clock times without explicit timezone localization.
Repeated or missing daylight-saving clock times therefore need explicit handling before production
deployment. Also, quarterly files are grouped by trip start: the first arrival bins may miss trips
started before the earliest downloaded quarter, while final-quarter trips can end after June 30.

## 7. Build the station-by-time panel

```bash
bikeflow prepare
```

For each station, the builder creates a regular 30-minute grid between its first and last observed
event. Missing event buckets inside those bounds are filled with zero. Filling these buckets is
necessary: omitting them would train the model primarily on windows in which a trip occurred.

The `observed_bounds` approach approximates historical station presence. It cannot distinguish a
temporary service closure from a genuinely quiet period, and the retrospective last-event boundary
must not be treated as an online operating-status signal.

The builder currently produces 5,798,715 rows in 180 station partitions:

```text
data/processed/panel/station_id=3005/panel.parquet
data/processed/stations.parquet
artifacts/prepare_summary.json
```

Partitioning makes preprocessing inspectable, but the current model trainer concatenates selected
partitions into memory. Full training therefore needs memory for several copies of the feature table.

## 8. Construct features with explicit time meaning

At a row's forecast origin `t`, `departures_lag_1` contains the completed bucket `[t-30m, t)`.
The current input bucket is indexed by its **end**, so the implementation uses `shift(lag - 1)`.
Using a zero shift for `lag_1` is correct under this indexing convention.

| Feature family | Exact content |
|---|---|
| Departure lags | 1, 2, 3, 48, 336, 672, 1008, 1344 half-hour offsets |
| Arrival lags | The same offsets for returns |
| Net-flow lags | Arrival count minus departure count at those offsets |
| Rolling sums | Previous 2, 6, 48, and 336 completed buckets |
| Station | Numeric ID, latitude, longitude, days since official launch |
| Calendar | Hour, weekday, weekend flag, month |
| Cycles | Hour and weekday sine/cosine encodings |

There are 48 numeric features in the saved run. A 336-bucket daily-aligned offset represents the
corresponding half-hour window one week earlier; 1344 represents four weeks earlier. Rolling windows
cover 1 hour, 3 hours, 24 hours, and 7 days.

The `hour` integer feature does not distinguish :00 from :30, but the sine/cosine hour encoding does.
Latitude and longitude describe station location. There is a nearest-station utility in `spatial.py`,
but neighbor demand aggregates are not currently connected to training. Inventory, weather, events,
and future trip characteristics are also absent from the fitted feature set.

## 9. Construct the future targets

For a row at 08:00:

```text
Known features:      events strictly before 08:00
30-minute target:   events during 08:00–08:30
60-minute target:   events during 08:00–09:00
```

The next bucket supplies the 30-minute target; the next two buckets are summed for 60 minutes.
Net-flow targets subtract departures from arrivals. Trailing rows without future coverage remain
missing, and training excludes rows missing the target or any required feature.

The longest lag removes roughly the first four weeks at each station. This also excludes short-lived
activity stations and limits coverage of newly launched stations. A cold-start model is future work.

## 10. Split historical periods

The configured chronological periods are:

| Split | Dates | Rows after required-feature filtering |
|---|---|---:|
| Training | 2024-04-01–2025-12-31 | 4,295,051 |
| Validation | 2026-01-01–2026-03-31 | 645,275 |
| Test | 2026-04-01–2026-06-30 | 639,897 |

A chronological holdout estimates performance on later dates. The test is a sequence of 30-minute
forecasts: later test rows may use real activity that occurred earlier in the test quarter, which
would already be known at their forecast origins. It is not a single three-month-ahead forecast.

Validation is separated by the repository, but the current fallback estimator does not use that
chronological validation frame for tuning. It uses scikit-learn's automatic internal validation
split. The saved model reports that internal early stopping was enabled and completed all 500
iterations. For a stricter time-series selection protocol, implement tuning and stopping explicitly
against the chronological validation period.

## 11. Train the model and calculate the seasonal baseline

```bash
bikeflow train
```

Do not add `--max-rows` for a full run. That argument is useful for debugging, but the loader reads
station files in sorted order and stops at the cap; it does not sample all stations evenly. The old
500,000-row development run consequently produced test predictions for only 12 stations.

The model builder first tries LightGBM with a Poisson objective. If the import fails, including a
missing macOS OpenMP library, it switches to scikit-learn histogram gradient boosting with Poisson
loss and prints a warning. That warning does not mean training failed.

The saved full run used the fallback with these effective parameters:

```text
loss = poisson
learning_rate = 0.05
max_iter = 500
max_leaf_nodes = 63
l2_regularization = 1.0
min_samples_leaf = 20
random_state = 42
```

Fallback defaults are not identical to LightGBM parameters. For example, `min_child_samples=40` in
the LightGBM configuration does not become the fallback's `min_samples_leaf`; the latter remained
20. Numeric station IDs are also not explicitly marked categorical in the fallback.

The seasonal baseline averages the previous four weekly counts. Its function currently uses
`336, 672, 1008, 1344`, irrespective of the extra daily lag listed in the baseline YAML configuration.
An always-zero baseline is additionally calculated in the report analysis to expose the effect of
the many zero-trip windows.

The nonnegative conditional mean is appropriate for count targets, but a Poisson objective does not
establish that counts are truly Poisson-distributed or provide calibrated prediction intervals.

## 12. Inspect predictions and measured results

```bash
python -m json.tool artifacts/training_summary.json
```

Training writes the current run into these fixed paths:

| File | Content |
|---|---|
| `artifacts/demand_model.joblib` | Estimator, features, target, backend, training timestamp |
| `artifacts/test_predictions.parquet` | Station, origin, actual target, model forecast, baseline |
| `artifacts/training_summary.json` | Feature list, split sizes, and aggregate metrics |

Repeated training overwrites these outputs. Archive them under a run-specific directory before
running another experiment. Current configs are not saved as immutable snapshots by the trainer.

To reproduce the expanded analysis without retraining:

```bash
python scripts/analyze_results.py
```

This reads the saved predictions and raw Q2 files, verifies that metrics match the training summary,
adds the zero baseline, calculates station/time slices, and writes `reports/full_run_analysis.json`.
It also calculates descriptive operating patterns independently of the trained-model predictions.

Optional figure reproduction:

```bash
python -m pip install -e '.[report]'
python scripts/plot_results.py
```

This adds Matplotlib and renders the report's three PNG figures from the analysis JSON. The existing
Markdown report is a dated narrative; recalculate and review its claims after each new model run.

## 13. Understand the evaluation

```text
MAE  = mean(abs(prediction - actual))
RMSE = sqrt(mean((prediction - actual)^2))
WAPE = sum(abs(prediction - actual)) / sum(actual)
Bias = mean(prediction - actual)
```

MAE and RMSE are in trips per station per 30 minutes. WAPE is relative error, not accuracy. Because
the same actual observations form the denominator, WAPE and MAE rank models identically on a fixed
evaluation set. RMSE emphasizes large misses; Bias measures systematic over- or under-prediction.

Always compare against zero and seasonal baselines, then examine zero versus positive windows,
weekday/weekend, hour, month, and station volume. The report assigns volume quartiles using training
traffic rather than using test outcomes to define a deployment segment.

## 14. View the Dashboard

```bash
streamlit run src/bikeflow/dashboard.py
```

Open `http://localhost:8501` and refresh after training. Press Ctrl+C in the server terminal to stop.
The Dashboard reads the saved prediction file, not raw data, and therefore now has 150 predicted
stations rather than the old 12-station subset.

Interpret the existing display carefully:

- It shows historical test forecasts, not predictions for today's next 30 minutes.
- It takes each station's last available row; stations can have different displayed origins.
- Map colors are relative demand quartiles, not empty/full risk probabilities. Rank-based quartiles
  can separate tied or nearly identical demand values into different colors.
- The separately collected status snapshot is not aligned to the historical forecast time.
- The table labeled Dispatch review sorts departures; it does not compute optimized bike transfers.

The map is useful for exploring model coverage. Inventory recommendations need a time-aligned view.

## 15. Translate trip patterns into operating hypotheses

The report aggregates raw departures and arrivals for fixed morning/evening windows. For example,
Union Station West Portal loses bikes through customer trips in weekday mornings and gains them in
weekday evenings. That supports checking bike supply before the morning peak and dock space before
evening returns.

These are historical associations, not evidence that a particular truck intervention will reduce
stockouts. The correct delivery quantity depends on opening inventory, arrival/departure sequence,
disabled bikes/docks, capacity, service hours, forecast uncertainty, and operator movements.

## 16. Collect inventory history for availability modeling

```bash
bikeflow collect-status
```

One call saves one snapshot; it does not start continuous collection. Schedule repeated execution,
for example every five minutes, to accumulate history. The collector saves raw GeoJSON and a
normalized Parquet table with collection time, bikes, docks, capacity, and status fields.

Before training an availability classifier, improve the preliminary label builder to verify actual
timestamp intervals, complete future coverage, operational status, stale records, and consecutive
zeros. It currently works by future row position and is not sufficient for irregular collection.
Then fit and calibrate separate empty/full classifiers and evaluate PR-AUC, Brier score, event
recall, and Recall@K. A config block alone does not implement those classifiers.

## 17. Compare retraining policies as a separate experiment

The backtest command exists, but a full strategy comparison has not been completed for this run.
For a development check confined to validation dates:

```bash
bikeflow backtest \
  --strategy fixed \
  --strategy weekly \
  --strategy monthly \
  --strategy drift_triggered \
  --evaluation-start 2026-01-01 \
  --evaluation-end "2026-03-31 23:59:59"
```

This evaluates weekly windows and, by default, fits from a trailing year capped at 750,000 rows.
It can be expensive. `--max-rows` limits the station coverage; do not label such a run a full-network
comparison. `--max-training-rows 0` removes the training-row cap if memory permits.

The current drift policy checks a rolling MAE ratio of 1.25 and feature PSI of 0.20, and applies the
decision at the next weekly origin. Not every setting in `drift.yaml` is connected to executable
logic. PSI indicates distribution change, not proof that the input-to-target relationship changed.
Before a final comparison, fix categorical handling on the LightGBM non-retraining path, validate
PSI on zero-heavy or constant reference data, and freeze thresholds using validation dates only.

## 18. Verify and maintain the repository

```bash
ruff check .
pytest
```

Tests check specific calculations and interfaces; passing them does not establish forecasting
accuracy, continuous collection, or safe deployment. Preserve data hashes, exact dependency versions,
frozen configs, code revision, models, and results for each serious experiment. The expanded analysis
records current artifact hashes and installed library versions so the report can be traced to the
saved full run.

The next model-development priorities are: a faithful chronological tuning procedure, a corrected
availability data contract, arrival and 60-minute training, spatial aggregates, uncertainty estimates,
and a time-aligned Dashboard. New claims should be evaluated on a later untouched period because
the existing test quarter has now been inspected repeatedly.
