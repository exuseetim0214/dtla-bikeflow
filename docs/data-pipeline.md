# Data pipeline

## Source registry

`configs/data_manifest.json` is the authoritative list of nine quarterly archives. Downloading is
idempotent: an existing file is reused unless `--force` is supplied. Every downloaded and extracted
file receives a SHA-256 digest in `data/raw/manifests/download_manifest.json`.

## Audit checks

`bikeflow audit` verifies:

- all 15 required trip columns exist;
- row count, timestamp coverage, missing values, and within-quarter duplicate IDs;
- station IDs can be joined to the official station table;
- DTLA starts, DTLA ends, and both cross-boundary directions are counted separately.

The audit is chunked, so it can run without loading all trips into memory.

## DTLA selection

The official station table `region == "DTLA"` is the primary definition. Coordinates are retained for
mapping and neighbor distance, not used as the authoritative boundary. Station 3000 is a virtual
operations station and is excluded from station-level demand targets.

## Panel time semantics

For a row with `forecast_time = t`:

- `departures_lag_1` is the observed count in `[t-30m, t)`;
- `departures_lag_2` is the count in `[t-60m, t-30m)`;
- `target_departures_30m` is the count in `[t, t+30m)`;
- `target_departures_60m` is the sum over `[t, t+60m)`.

Rolling features include only completed windows. The targets are shifted toward the future only after
features are constructed. Tests assert this contract.

## Station presence

The current official station table does not provide historical close dates. The default
`observed_bounds` policy creates rows between each station's first and last observed trip event. It
avoids fabricating zeros before launch or after a station disappears, but it cannot identify temporary
closures. Historical status snapshots should replace this approximation when available.

