from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schemas import Constraint, PromptRecord


def load_jsonl_dataset(path: Path) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            data = json.loads(raw)
            constraints = [
                Constraint(
                    type=item["type"],
                    value=item["value"],
                    category=item.get("category", "uncategorized"),
                )
                for item in data.get("constraints", [])
            ]
            records.append(
                PromptRecord(
                    prompt_id=data["prompt_id"],
                    original_prompt=data["prompt"],
                    constraints=constraints,
                    ideal_response=data.get("ideal_response"),
                    metadata={"line_number": line_number},
                )
            )
    return records
