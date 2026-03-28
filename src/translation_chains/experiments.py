from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .adapters import make_model_adapter, make_translation_adapter
from .datasets import load_dataset_from_config
from .evaluators import evaluate_record
from .reporting import build_reports
from .schemas import ExperimentStatus, TranslationStep
from .status import write_status


def run_experiment(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    output_dir = _resolve(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_from_config(config_path, config)
    _validate_dataset_support(dataset, config)
    translation = make_translation_adapter(config["translation_engine"])
    models = [make_model_adapter(name) for name in config["models"]]
    status_path = output_dir / "status.json"
    run_name = config.get("run_name", output_dir.name)
    dataset_source = _describe_dataset_source(config)
    expected_rows = _expected_row_count(dataset_size=len(dataset), config=config)
    status = ExperimentStatus(
        run_name=run_name,
        status="running",
        dataset_source=dataset_source,
        dataset_size=len(dataset),
        expected_rows=expected_rows,
        completed_rows=0,
        started_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(output_dir),
        notes=[],
    )
    write_status(status_path, status)

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as sink:
        for record in dataset:
            for model in models:
                baseline = evaluate_record(
                    record=record,
                    model_name=model.name,
                    regime="baseline",
                    language_path=["en"],
                    depth=0,
                    evaluated_prompt=record.original_prompt,
                    raw_response=model.generate(record, record.original_prompt),
                )
                baseline.extra["dataset_source"] = dataset_source
                sink.write(json.dumps(asdict(baseline), ensure_ascii=False) + "\n")
                status.completed_rows += 1
                status.updated_at = datetime.now(timezone.utc).isoformat()
                write_status(status_path, status)

                for regime in config["regimes"]:
                    for path in config["paths"]:
                        translated_prompt = record.original_prompt
                        history: List[TranslationStep] = []
                        for depth, (source_lang, target_lang) in enumerate(
                            zip(path, path[1:]),
                            start=1,
                        ):
                            translated_prompt = translation.translate(
                                translated_prompt,
                                source_lang,
                                target_lang,
                            )
                            history.append(
                                TranslationStep(
                                    step_index=depth,
                                    source_lang=source_lang,
                                    target_lang=target_lang,
                                    input_text=record.original_prompt if depth == 1 else history[-1].output_text,
                                    output_text=translated_prompt,
                                    engine_name=translation.name,
                                )
                            )

                            evaluated_prompt = translated_prompt
                            if regime == "back_translated" and target_lang != "en":
                                evaluated_prompt = translation.translate(
                                    translated_prompt,
                                    target_lang,
                                    "en",
                                )

                            response = model.generate(record, evaluated_prompt)
                            result = evaluate_record(
                                record=record,
                                model_name=model.name,
                                regime=regime,
                                language_path=path[: depth + 1],
                                depth=depth,
                                evaluated_prompt=evaluated_prompt,
                                raw_response=response,
                            )
                            result.extra["dataset_source"] = dataset_source
                            result.extra["translation_step"] = asdict(history[-1])
                            sink.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                            status.completed_rows += 1
                            status.updated_at = datetime.now(timezone.utc).isoformat()
                            write_status(status_path, status)

    status.status = "completed"
    status.finished_at = datetime.now(timezone.utc).isoformat()
    status.updated_at = status.finished_at
    write_status(status_path, status)
    build_reports(results_path, output_dir)
    return results_path


def _resolve(config_path: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def _describe_dataset_source(config: Dict[str, Any]) -> str:
    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "local_jsonl")
    if dataset_type == "huggingface":
        return f"huggingface:{dataset_cfg.get('name', 'unknown')}:{dataset_cfg.get('split', 'train')}"
    if dataset_type == "local_jsonl":
        return f"local_jsonl:{dataset_cfg.get('path', 'unknown')}"
    return dataset_type


def _expected_row_count(dataset_size: int, config: Dict[str, Any]) -> int:
    path_steps = sum(max(len(path) - 1, 0) for path in config["paths"])
    per_model = 1 + (len(config["regimes"]) * path_steps)
    return dataset_size * len(config["models"]) * per_model


def _validate_dataset_support(dataset: List[Any], config: Dict[str, Any]) -> None:
    allow_placeholder = bool(config.get("allow_placeholder_ifeval_evaluator", False))
    has_placeholder_ifeval = any(
        constraint.type == "ifeval_instruction"
        for record in dataset
        for constraint in getattr(record, "constraints", [])
    )
    if has_placeholder_ifeval and not allow_placeholder:
        raise ValueError(
            "This dataset appears to require a real IFEval-style validator, but the current "
            "repository only has a placeholder evaluator for raw instruction IDs. "
            "Set `allow_placeholder_ifeval_evaluator` to true only for debugging, or integrate "
            "the official validator before treating the results as benchmark-valid."
        )
