from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_results
from .datasets import load_jsonl_dataset
from .datasets import load_dataset_from_config
from .experiments import run_experiment
from .reporting import build_reports
from .status import read_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-chains")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a JSON config.")
    run_parser.add_argument("--config", required=True, help="Path to runtime config JSON.")

    report_parser = subparsers.add_parser("report", help="Rebuild reports from results JSONL.")
    report_parser.add_argument("--results", required=True, help="Path to results JSONL.")
    report_parser.add_argument("--output-dir", required=True, help="Directory for tables and summaries.")

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Print a dataset summary.")
    inspect_parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset.")

    inspect_config_parser = subparsers.add_parser("inspect-config", help="Print dataset and run info from a config.")
    inspect_config_parser.add_argument("--config", required=True, help="Path to runtime config JSON.")

    status_parser = subparsers.add_parser("status", help="Inspect experiment progress from status.json.")
    status_parser.add_argument("--status-file", required=True, help="Path to status.json.")

    audit_parser = subparsers.add_parser("audit-results", help="Check whether results look complete and reasonable.")
    audit_parser.add_argument("--results", required=True, help="Path to results JSONL.")
    audit_parser.add_argument("--output-dir", required=True, help="Directory for audit output.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config_path = Path(args.config)
        run_experiment(config_path)
        return 0

    if args.command == "report":
        build_reports(Path(args.results), Path(args.output_dir))
        return 0

    if args.command == "inspect-dataset":
        dataset = load_jsonl_dataset(Path(args.dataset))
        summary = {
            "records": len(dataset),
            "prompt_ids": [item.prompt_id for item in dataset],
            "categories": sorted(
                {
                    constraint.category
                    for item in dataset
                    for constraint in item.constraints
                    if constraint.category
                }
            ),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "inspect-config":
        config_path = Path(args.config)
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        dataset = load_dataset_from_config(config_path, config)
        dataset_cfg = config.get("dataset", {})
        summary = {
            "dataset_type": dataset_cfg.get("type", "local_jsonl"),
            "dataset_name": dataset_cfg.get("name"),
            "dataset_split": dataset_cfg.get("split"),
            "records": len(dataset),
            "models": [
                item if isinstance(item, str) else item.get("model_id", item.get("name"))
                for item in config.get("models", [])
            ],
            "paths": config.get("paths", []),
            "regimes": config.get("regimes", []),
            "translation_engine": config.get("translation_engine"),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "status":
        status = read_status(Path(args.status_file))
        print(json.dumps(status.__dict__, indent=2))
        return 0

    if args.command == "audit-results":
        audit_path = audit_results(Path(args.results), Path(args.output_dir))
        print(json.dumps({"audit_path": str(audit_path)}, indent=2))
        return 0

    parser.error("Unknown command")
    return 1
