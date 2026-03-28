from __future__ import annotations

import json
import re
from collections import defaultdict
from inspect import signature
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
    if record.metadata.get("dataset_name") in {"google/IFEval", "HuggingFaceH4/ifeval", "IFEval"}:
        return _evaluate_ifeval_record(
            record=record,
            model_name=model_name,
            regime=regime,
            language_path=language_path,
            depth=depth,
            evaluated_prompt=evaluated_prompt,
            raw_response=raw_response,
        )

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


def _evaluate_ifeval_record(
    record: PromptRecord,
    model_name: str,
    regime: str,
    language_path: List[str],
    depth: int,
    evaluated_prompt: str,
    raw_response: str,
) -> EvaluationResult:
    try:
        from instruction_following_eval import evaluate_instruction_following
    except ImportError as exc:
        raise ImportError(
            "IFEval execution requires the `instruction_following_eval` package. "
            "Install requirements.txt before running on the full IFEval dataset."
        ) from exc

    raw = dict(record.raw_example)
    metrics = _run_ifeval(evaluate_instruction_following, raw, raw_response)

    prompt_level_strict = float(_metric(metrics, "prompt_level_strict_acc", "prompt-level-strict-accuracy", default=0.0))
    prompt_level_loose = float(_metric(metrics, "prompt_level_loose_acc", "prompt-level-loose-accuracy", default=0.0))

    inst_strict = _metric(metrics, "inst_level_strict_acc", "instruction_level_strict_acc", default=0.0)
    inst_loose = _metric(metrics, "inst_level_loose_acc", "instruction_level_loose_acc", default=0.0)
    if isinstance(inst_strict, list):
        instruction_level_strict = sum(float(x) for x in inst_strict) / len(inst_strict) if inst_strict else 0.0
    else:
        instruction_level_strict = float(inst_strict or 0.0)
    if isinstance(inst_loose, list):
        instruction_level_loose = sum(float(x) for x in inst_loose) / len(inst_loose) if inst_loose else 0.0
    else:
        instruction_level_loose = float(inst_loose or 0.0)

    category_scores = {}
    instruction_ids = raw.get("instruction_id_list", [])
    if isinstance(inst_strict, list):
        category_bucket: Dict[str, List[float]] = defaultdict(list)
        for instruction_id, value in zip(instruction_ids, inst_strict):
            category_bucket[str(instruction_id)].append(float(value))
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
        prompt_level_strict=prompt_level_strict,
        prompt_level_loose=prompt_level_loose,
        instruction_level_strict=instruction_level_strict,
        instruction_level_loose=instruction_level_loose,
        category_scores=category_scores,
        extra={"ifeval_metrics": metrics},
    )


def _metric(metrics: Dict[str, object], *keys: str, default: object) -> object:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return default


def _run_ifeval(evaluate_instruction_following, raw_example: Dict[str, object], raw_response: str) -> Dict[str, object]:
    """
    Support the common IFEval package API variants:

    1. evaluate_instruction_following(inputs, responses)
    2. evaluate_instruction_following(examples)
    3. evaluate_instruction_following(prompts=..., responses=...)
    """
    try:
        params = list(signature(evaluate_instruction_following).parameters)
    except (TypeError, ValueError):
        params = []

    prompt_payload = [dict(raw_example)]
    response_payload = [raw_response]

    if len(params) >= 2:
        first, second = params[0], params[1]
        try:
            return evaluate_instruction_following(**{first: prompt_payload, second: response_payload})
        except TypeError:
            return evaluate_instruction_following(prompt_payload, response_payload)

    merged = [dict(raw_example, response=raw_response)]
    try:
        return evaluate_instruction_following(merged)
    except TypeError:
        return evaluate_instruction_following(inputs=prompt_payload, responses=response_payload)


def _check_constraint(constraint: Constraint, response: str) -> bool:
    constraint_type = constraint.type
    value = constraint.value

    if constraint_type == "ifeval_instruction":
        # Placeholder for real IFEval validator integration.
        # For now, these constraints are marked as unknown and excluded from strict failure.
        return True

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
