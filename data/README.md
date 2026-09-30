# Data layout

All source files are public Metro Bike Share data. Raw files and generated tables are intentionally
excluded from Git because they are large and reproducible.

```text
data/
├── raw/
│   ├── archives/       downloaded quarterly ZIP files
│   ├── trips/          one extracted CSV per quarter
│   ├── stations/       official station table
│   └── manifests/      byte size and SHA-256 download audit
├── interim/            cleaned trips and station lookup
├── processed/          model-ready station × 30-minute partitions
└── status/             timestamped live-status snapshots
```

Run `make download` to reproduce `data/raw`. The source registry lives in
[`configs/data_manifest.json`](../configs/data_manifest.json).

The repository never treats the live status feed as historical data. `make collect-status` stores a
new timestamped snapshot; repeated collection is required before real empty/full labels can be built.

