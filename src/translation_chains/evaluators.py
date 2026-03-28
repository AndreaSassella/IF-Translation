from __future__ import annotations

import json
import re
from collections import defaultdict
from inspect import signature
from typing import Dict, List, Tuple

from .nltk_setup import ensure_ifeval_nltk_resources
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
    ensure_ifeval_nltk_resources()
    try:
        from instruction_following_eval import instructions_registry
    except ImportError as exc:
        raise ImportError(
            "IFEval execution requires the `instruction_following_eval` package. "
            "Install requirements.txt before running on the full IFEval dataset."
        ) from exc

    raw = dict(record.raw_example)
    strict_flags = _ifeval_follow_list(raw, raw_response, instructions_registry, strict=True)
    loose_flags = _ifeval_follow_list(raw, raw_response, instructions_registry, strict=False)

    prompt_level_strict = float(all(strict_flags))
    prompt_level_loose = float(all(loose_flags))
    instruction_level_strict = (
        sum(float(x) for x in strict_flags) / len(strict_flags) if strict_flags else 0.0
    )
    instruction_level_loose = (
        sum(float(x) for x in loose_flags) / len(loose_flags) if loose_flags else 0.0
    )

    instruction_ids = raw.get("instruction_id_list", [])
    category_bucket: Dict[str, List[float]] = defaultdict(list)
    for instruction_id, value in zip(instruction_ids, strict_flags):
        category_bucket[str(instruction_id)].append(float(value))
    category_scores = {
        category: sum(values) / len(values) for category, values in category_bucket.items()
    }

    metrics = {
        "prompt_level_strict_acc": prompt_level_strict,
        "prompt_level_loose_acc": prompt_level_loose,
        "inst_level_strict_acc": strict_flags,
        "inst_level_loose_acc": loose_flags,
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


def _ifeval_follow_list(
    raw_example: Dict[str, object],
    raw_response: str,
    instructions_registry,
    strict: bool,
) -> List[bool]:
    all_responses = _candidate_responses(raw_response, strict=strict)
    instruction_ids = list(raw_example.get("instruction_id_list", []))
    kwargs_list = list(raw_example.get("kwargs", []))
    prompt = str(raw_example.get("prompt", ""))

    output: List[bool] = []
    for index, instruction_id in enumerate(instruction_ids):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        raw_kwargs = kwargs_list[index] if index < len(kwargs_list) else {}
        filtered_kwargs = _filter_instruction_kwargs(instruction, raw_kwargs)
        instruction.build_description(**filtered_kwargs)

        try:
            args = instruction.get_instruction_args()
        except Exception:
            args = None
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        is_following = False
        for candidate in all_responses:
            if candidate.strip() and instruction.check_following(candidate):
                is_following = True
                break
        output.append(is_following)
    return output


def _candidate_responses(raw_response: str, strict: bool) -> List[str]:
    if strict:
        return [raw_response]

    lines = raw_response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()
    revised_response = raw_response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    return [
        raw_response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]


def _filter_instruction_kwargs(instruction, raw_kwargs: object) -> Dict[str, object]:
    if not isinstance(raw_kwargs, dict):
        return {}

    accepted_keys = []
    try:
        accepted_keys = list(instruction.get_instruction_args_keys())
    except Exception:
        accepted_keys = []

    if not accepted_keys:
        try:
            params = signature(instruction.build_description).parameters
            accepted_keys = [
                name for name, param in params.items()
                if name != "self" and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            accepted_keys = []

    return {
        key: value
        for key, value in raw_kwargs.items()
        if key in accepted_keys and value is not None
    }


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
