from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from .schemas import Constraint, EvaluationResult, PromptRecord


def evaluate_record(
    record: PromptRecord,
    model_name: str,
    regime: str,
    language_path: List[str],
    depth: int,
    evaluated_prompt: str,
    raw_response: str,
) -> EvaluationResult:
    checks = []
    category_bucket: Dict[str, List[float]] = defaultdict(list)

    for constraint in record.constraints:
        passed = float(_check_constraint(constraint, raw_response))
        checks.append(passed)
        category_bucket[constraint.category].append(passed)

    instruction_level = sum(checks) / len(checks) if checks else 1.0
    prompt_level = 1.0 if all(check == 1.0 for check in checks) else 0.0
    category_scores = {
        category: sum(values) / len(values) for category, values in category_bucket.items()
    }

    return EvaluationResult(
        model_name=model_name,
        prompt_id=record.prompt_id,
        regime=regime,
        language_path=language_path,
        depth=depth,
        evaluated_prompt=evaluated_prompt,
        raw_response=raw_response,
        prompt_level_strict=prompt_level,
        prompt_level_loose=instruction_level,
        instruction_level_strict=instruction_level,
        instruction_level_loose=instruction_level,
        category_scores=category_scores,
        extra={"constraint_checks": checks},
    )


def _check_constraint(constraint: Constraint, response: str) -> bool:
    constraint_type = constraint.type
    value = constraint.value

    if constraint_type == "bullet_count":
        bullets = [line for line in response.splitlines() if line.strip().startswith("- ")]
        return len(bullets) == int(value)

    if constraint_type == "sentence_count":
        sentences = [part for part in re.split(r"[.!?]+", response) if part.strip()]
        return len(sentences) == int(value)

    if constraint_type == "must_contain":
        return str(value).lower() in response.lower()

    if constraint_type == "forbidden_word":
        return str(value).lower() not in response.lower()

    if constraint_type == "starts_with":
        first_line = response.strip().splitlines()[0] if response.strip() else ""
        return first_line.startswith(str(value))

    if constraint_type == "paragraph_max_words":
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        paragraph = lines[1] if len(lines) > 1 else ""
        return len(paragraph.split()) <= int(value)

    if constraint_type == "json_keys":
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return False
        return all(key in parsed for key in value)

    if constraint_type == "comma_separated_count":
        parts = [item.strip() for item in response.split(",") if item.strip()]
        return len(parts) == int(value)

    return False
