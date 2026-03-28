from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def audit_results(results_path: Path, output_dir: Path) -> Path:
    rows = list(_load_jsonl(results_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    findings = []
    if not rows:
        findings.append({"level": "error", "message": "No result rows found."})
    else:
        findings.extend(_check_score_ranges(rows))
        findings.extend(_check_baselines(rows))
        findings.extend(_check_reasonable_ordering(rows))
        findings.extend(_check_regimes_present(rows))

    audit = {
        "results_path": str(results_path),
        "row_count": len(rows),
        "status": "pass" if not any(item["level"] == "error" for item in findings) else "fail",
        "findings": findings,
    }

    out_path = output_dir / "audit.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    return out_path


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _check_score_ranges(rows: List[dict]) -> List[dict]:
    findings = []
    for metric in ("prompt_level_strict", "instruction_level_strict"):
        bad = [
            row for row in rows
            if row.get(metric) is not None and not (0.0 <= float(row[metric]) <= 1.0)
        ]
        if bad:
            findings.append(
                {
                    "level": "error",
                    "message": f"Metric {metric} has values outside [0, 1].",
                    "count": len(bad),
                }
            )
    return findings


def _check_baselines(rows: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model_name"]].append(row)

    findings = []
    for model_name, model_rows in grouped.items():
        if not any(row["regime"] == "baseline" and row["depth"] == 0 for row in model_rows):
            findings.append(
                {
                    "level": "error",
                    "message": f"Model {model_name} has no baseline rows.",
                }
            )
    return findings


def _check_reasonable_ordering(rows: List[dict]) -> List[dict]:
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["model_name"]][row["depth"]].append(row["instruction_level_strict"])

    findings = []
    for model_name, depth_map in grouped.items():
        if 0 not in depth_map:
            continue
        baseline = sum(depth_map[0]) / len(depth_map[0])
        worst_depth = min(sum(values) / len(values) for depth, values in depth_map.items() if depth != 0) if any(depth != 0 for depth in depth_map) else baseline
        if worst_depth > baseline + 1e-9:
            findings.append(
                {
                    "level": "warning",
                    "message": f"Model {model_name} performs better at perturbed depths than at baseline on average.",
                }
            )
    return findings


def _check_regimes_present(rows: List[dict]) -> List[dict]:
    regimes = {row["regime"] for row in rows}
    findings = []
    for expected in ("baseline", "native", "back_translated"):
        if expected not in regimes:
            findings.append(
                {
                    "level": "warning",
                    "message": f"Regime {expected} is missing from the results.",
                }
            )
    return findings
