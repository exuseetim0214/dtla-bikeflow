# Documentation

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | Components, storage layers, and online/offline flow |
| [Data pipeline](data-pipeline.md) | Download, validation, DTLA joins, panel construction |
| [Modeling](modeling.md) | Targets, features, baselines, LightGBM, leakage controls |
| [Availability risk](availability-risk.md) | Status collection and empty/full labels |
| [Evaluation](evaluation.md) | Splits, metrics, slices, and acceptance criteria |
| [Concept drift](drift.md) | Detection and retraining comparison |
| [Dashboard](dashboard.md) | Map, risk presentation, and dispatch ranking |

Configuration is executable documentation. Data lineage lives in `configs/data_manifest.json`, time
semantics in `configs/pipeline.yaml`, model parameters in `configs/model.yaml`, and drift thresholds
in `configs/drift.yaml`.

