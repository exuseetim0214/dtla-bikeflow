from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any


def _print_result(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _download(args: argparse.Namespace) -> None:
    from bikeflow.download import download_dataset

    _print_result(download_dataset(args.manifest, args.raw_dir, force=args.force))


def _audit(args: argparse.Namespace) -> None:
    from bikeflow.audit import audit_dataset

    _print_result(audit_dataset(args.raw_dir, args.output_dir))


def _prepare(args: argparse.Namespace) -> None:
    from bikeflow.prepare import prepare_dataset

    _print_result(prepare_dataset(args.config))


def _train(args: argparse.Namespace) -> None:
    from bikeflow.modeling import train_demand_model

    _print_result(
        train_demand_model(args.pipeline_config, args.model_config, max_rows=args.max_rows)
    )


def _collect_status(args: argparse.Namespace) -> None:
    from bikeflow.status import collect_status_snapshot

    _print_result(collect_status_snapshot(args.url, args.output_dir))


def _backtest(args: argparse.Namespace) -> None:
    from bikeflow.backtest import rolling_retraining_backtest

    _print_result(
        rolling_retraining_backtest(
            args.pipeline_config,
            args.model_config,
            args.drift_config,
            strategies=args.strategy,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            training_window_days=args.training_window_days,
            max_rows=args.max_rows,
            max_training_rows=args.max_training_rows,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bikeflow", description="DTLA BikeFlow pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and verify official Metro data")
    download.add_argument("--manifest", default="configs/data_manifest.json")
    download.add_argument("--raw-dir", default="data/raw")
    download.add_argument("--force", action="store_true")
    download.set_defaults(handler=_download)

    audit = subparsers.add_parser("audit", help="Audit downloaded trip and station files")
    audit.add_argument("--raw-dir", default="data/raw")
    audit.add_argument("--output-dir", default="reports")
    audit.set_defaults(handler=_audit)

    prepare = subparsers.add_parser("prepare", help="Build station × 30-minute feature partitions")
    prepare.add_argument("--config", default="configs/pipeline.yaml")
    prepare.set_defaults(handler=_prepare)

    train = subparsers.add_parser("train", help="Train and evaluate the 30-minute demand model")
    train.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    train.add_argument("--model-config", default="configs/model.yaml")
    train.add_argument("--max-rows", type=int)
    train.set_defaults(handler=_train)

    backtest = subparsers.add_parser("backtest", help="Compare rolling retraining strategies")
    backtest.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    backtest.add_argument("--model-config", default="configs/model.yaml")
    backtest.add_argument("--drift-config", default="configs/drift.yaml")
    backtest.add_argument(
        "--strategy",
        action="append",
        choices=["fixed", "weekly", "monthly", "drift_triggered"],
    )
    backtest.add_argument("--evaluation-start")
    backtest.add_argument("--evaluation-end")
    backtest.add_argument("--training-window-days", type=int, default=365)
    backtest.add_argument("--max-rows", type=int)
    backtest.add_argument("--max-training-rows", type=int, default=750000)
    backtest.set_defaults(handler=_backtest)

    status = subparsers.add_parser("collect-status", help="Save one live station-status snapshot")
    status.add_argument("--url", default="https://bikeshare.metro.net/stations/json/")
    status.add_argument("--output-dir", default="data/status")
    status.set_defaults(handler=_collect_status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler: Callable[[argparse.Namespace], None] = args.handler
    handler(args)


if __name__ == "__main__":
    main()
