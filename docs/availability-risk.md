# Empty/full station risk

## Why trips are insufficient

Inventory evolves through rentals, returns, maintenance, and operator rebalancing. Only the first two
appear in public trip records. Therefore cumulative trip net flow is not a ground-truth inventory
series and must not be labeled as an empty/full probability.

## Collection contract

Run `bikeflow collect-status` every five minutes. Each normalized record contains collection time,
station ID, coordinates, available bikes, available docks, total docks, public status, connection
status, and virtual/event flags. Raw JSON is retained beside Parquet.

Monitor collection completeness:

- expected snapshots per station per day: 288;
- alert when a station has less than 95% coverage;
- exclude disconnected or non-public stations from labels;
- treat capacity changes as state changes, not ordinary demand.

## Labels

An empty/full event requires two consecutive zero snapshots to reduce one-snapshot noise. For a
30-minute horizon at five-minute sampling, inspect the next six snapshots. Do not create a negative
label when future snapshot coverage is incomplete.

## Classification features and metrics

Use current inventory, inventory ratio, capacity, predicted departures/arrivals, recent flow, station,
calendar, and neighbor pressure. Report PR-AUC, Brier score, calibration curves, event recall, and
Recall@K for the number of stations operations can actually service.

