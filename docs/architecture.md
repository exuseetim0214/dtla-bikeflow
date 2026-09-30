# Architecture

## Offline training path

```mermaid
flowchart TD
    A[Official quarterly ZIPs] --> B[Atomic downloader]
    B --> C[SHA-256 manifest]
    B --> D[Quarterly CSVs]
    E[Official Station Table] --> F[Station normalization]
    D --> G[Chunked flow aggregation]
    F --> G
    G --> H[Per-station 30-minute Parquet]
    H --> I[Time split]
    I --> J[Seasonal baselines]
    I --> K[Poisson demand model]
    J --> L[Comparable holdout report]
    K --> L
```

Trip files are read in chunks. The pipeline does not build one giant trip DataFrame. Departures and
arrivals are aggregated separately before a dense panel is produced for each observed DTLA station.
Per-station Parquet partitions keep feature generation and downstream loading inspectable.

## Online status path

```mermaid
flowchart LR
    A[Metro live GeoJSON] -->|every 5 min| B[Raw snapshot]
    B --> C[Normalized status Parquet]
    C --> D[Persistent zero labels]
    C --> E[Current inventory]
    D --> F[Availability classifier]
    G[Demand forecast] --> H[Expected lost trips]
    F --> H
    E --> H
    H --> I[Dispatch queue]
```

The status collector deliberately keeps both the source JSON and a normalized table. This supports
schema audits and prevents a future parser change from destroying the original evidence.

## Storage contracts

| Layer | Contract | Git policy |
|---|---|---|
| `data/raw` | Immutable downloaded bytes and extraction manifest | ignored |
| `data/interim` | Cleaned or normalized temporary data | ignored |
| `data/processed` | Model-ready station panel | ignored |
| `data/status` | Timestamped inventory snapshots | ignored |
| `artifacts` | Models, predictions, and metrics | ignored |
| `reports` | Small human-readable audits | committed |

