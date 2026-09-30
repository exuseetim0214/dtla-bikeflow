# Experiments

Each run should create a directory containing:

```text
experiments/<run-name>/
├── pipeline.yaml
├── model.yaml
├── data_manifest.json
├── metrics.json
├── feature_importance.csv
└── notes.md
```

Do not overwrite previous runs. Copy the exact configs and source data manifest used by the run. Test
metrics should be generated only after model selection is frozen on validation data.

