from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                    raw_example=dict(data),
                )
            )
    return records


def load_dataset_from_config(config_path: Path, config: Dict[str, Any]) -> List[PromptRecord]:
    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "local_jsonl")

    if dataset_type == "local_jsonl":
        path = _resolve(config_path, dataset_cfg["path"])
        return load_jsonl_dataset(path)

    if dataset_type == "huggingface":
        return load_huggingface_dataset(
            dataset_name=dataset_cfg["name"],
            split=dataset_cfg.get("split", "train"),
            prompt_field=dataset_cfg.get("prompt_field", "prompt"),
            prompt_id_field=dataset_cfg.get("prompt_id_field", "prompt_id"),
            constraints_field=dataset_cfg.get("constraints_field"),
            ideal_response_field=dataset_cfg.get("ideal_response_field"),
            limit=dataset_cfg.get("limit"),
        )

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    prompt_field: str = "prompt",
    prompt_id_field: str = "prompt_id",
    constraints_field: Optional[str] = None,
    ideal_response_field: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[PromptRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the `datasets` package. "
            "Install requirements.txt before using dataset.type='huggingface'."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records: List[PromptRecord] = []
    for index, item in enumerate(dataset):
        prompt_text = item.get(prompt_field)
        if isinstance(prompt_text, list):
            prompt_text = "\n".join(str(part) for part in prompt_text)

        prompt_id = item.get(prompt_id_field, f"{dataset_name}-{split}-{index}")
        constraints = _extract_constraints(item, constraints_field)
        ideal_response = item.get(ideal_response_field) if ideal_response_field else None
        records.append(
            PromptRecord(
                prompt_id=str(prompt_id),
                original_prompt=str(prompt_text),
                constraints=constraints,
                ideal_response=str(ideal_response) if ideal_response is not None else None,
                metadata={
                    "dataset_name": dataset_name,
                    "dataset_split": split,
                    "source": "huggingface",
                },
                raw_example=dict(item),
            )
        )
    return records


def _extract_constraints(item: Dict[str, Any], constraints_field: Optional[str]) -> List[Constraint]:
    if constraints_field == "instruction_id_list" and constraints_field in item:
        return [
            Constraint(type="ifeval_instruction", value=str(value), category="ifeval_instruction")
            for value in item[constraints_field]
        ]

    if constraints_field and constraints_field in item:
        raw_constraints = item[constraints_field]
        if isinstance(raw_constraints, list):
            output = []
            for index, raw in enumerate(raw_constraints):
                if isinstance(raw, dict):
                    output.append(
                        Constraint(
                            type=str(raw.get("type", f"constraint_{index}")),
                            value=raw.get("value"),
                            category=str(raw.get("category", "uncategorized")),
                        )
                    )
                else:
                    output.append(
                        Constraint(
                            type=f"constraint_{index}",
                            value=str(raw),
                            category="uncategorized",
                        )
                    )
            return output

    if "constraints" in item and isinstance(item["constraints"], list):
        return _extract_constraints(item, "constraints")

    if "instruction_id_list" in item and isinstance(item["instruction_id_list"], list):
        return [
            Constraint(type=str(value), value=True, category="ifeval_instruction")
            for value in item["instruction_id_list"]
        ]

    return []


def _resolve(config_path: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path
