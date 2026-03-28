from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import load_jsonl_dataset
from .experiments import run_experiment
from .reporting import build_reports


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

    parser.error("Unknown command")
    return 1
