from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .audit import audit_results


def build_reports(results_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(_load_jsonl(results_path))
    _write_summary_json(rows, output_dir / "summary.json")
    _write_model_depth_table(rows, output_dir / "model_by_depth.csv")
    _write_regime_table(rows, output_dir / "regime_summary.csv")
    _write_category_table(rows, output_dir / "category_scores.csv")
    _write_markdown_report(rows, output_dir / "REPORT.md")
    audit_results(results_path, output_dir)


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_summary_json(rows: List[dict], path: Path) -> None:
    unique_keys = {
        (
            row["model_name"],
            row["prompt_id"],
            row["regime"],
            tuple(row["language_path"]),
            row["depth"],
        )
        for row in rows
    }
    summary = {
        "rows": len(rows),
        "unique_rows": len(unique_keys),
        "duplicate_rows": len(rows) - len(unique_keys),
        "models": sorted({row["model_name"] for row in rows}),
        "regimes": sorted({row["regime"] for row in rows}),
        "depths": sorted({row["depth"] for row in rows}),
        "prompts": sorted({row["prompt_id"] for row in rows}),
        "dataset_sources": sorted(
            {
                row.get("extra", {}).get("dataset_source", "unknown")
                for row in rows
            }
        ),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _write_model_depth_table(rows: List[dict], path: Path) -> None:
    bucket: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (row["model_name"], row["depth"])
        bucket[key].append(row["instruction_level_strict"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "depth", "avg_instruction_level_strict"])
        for (model_name, depth) in sorted(bucket):
            values = bucket[(model_name, depth)]
            writer.writerow([model_name, depth, round(sum(values) / len(values), 4)])


def _write_regime_table(rows: List[dict], path: Path) -> None:
    bucket: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (row["model_name"], row["regime"])
        bucket[key].append(row["prompt_level_strict"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "regime", "avg_prompt_level_strict"])
        for (model_name, regime) in sorted(bucket):
            values = bucket[(model_name, regime)]
            writer.writerow([model_name, regime, round(sum(values) / len(values), 4)])


def _write_category_table(rows: List[dict], path: Path) -> None:
    bucket: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        for category, score in row.get("category_scores", {}).items():
            key = (row["model_name"], row["depth"], category)
            bucket[key].append(score)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "depth", "category", "avg_score"])
        for (model_name, depth, category) in sorted(bucket):
            values = bucket[(model_name, depth, category)]
            writer.writerow([model_name, depth, category, round(sum(values) / len(values), 4)])


def _write_markdown_report(rows: List[dict], path: Path) -> None:
    model_depth = _collect_model_depth(rows)
    regime_summary = _collect_regime_summary(rows)

    lines = [
        "# Translation Chains Report",
        "",
        "## Model by Depth",
        "",
        "| Model | Depth | Avg Instruction Score |",
        "|---|---:|---:|",
    ]
    for row in model_depth:
        lines.append(
            f"| {row['model_name']} | {row['depth']} | {row['avg_instruction_level_strict']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Regime Summary",
            "",
            "| Model | Regime | Avg Prompt Score |",
            "|---|---|---:|",
        ]
    )
    for row in regime_summary:
        lines.append(
            f"| {row['model_name']} | {row['regime']} | {row['avg_prompt_level_strict']:.4f} |"
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _collect_model_depth(rows: List[dict]) -> List[dict]:
    bucket: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (row["model_name"], row["depth"])
        bucket[key].append(row["instruction_level_strict"])
    output = []
    for (model_name, depth) in sorted(bucket):
        values = bucket[(model_name, depth)]
        output.append(
            {
                "model_name": model_name,
                "depth": depth,
                "avg_instruction_level_strict": sum(values) / len(values),
            }
        )
    return output


def _collect_regime_summary(rows: List[dict]) -> List[dict]:
    bucket: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (row["model_name"], row["regime"])
        bucket[key].append(row["prompt_level_strict"])
    output = []
    for (model_name, regime) in sorted(bucket):
        values = bucket[(model_name, regime)]
        output.append(
            {
                "model_name": model_name,
                "regime": regime,
                "avg_prompt_level_strict": sum(values) / len(values),
            }
        )
    return output
