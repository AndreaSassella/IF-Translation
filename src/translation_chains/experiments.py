from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .adapters import make_model_adapter, make_translation_adapter
from .datasets import load_jsonl_dataset
from .evaluators import evaluate_record
from .reporting import build_reports
from .schemas import TranslationStep


def run_experiment(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset_path = _resolve(config_path, config["dataset_path"])
    output_dir = _resolve(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl_dataset(dataset_path)
    translation = make_translation_adapter(config["translation_engine"])
    models = [make_model_adapter(name) for name in config["models"]]

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
                sink.write(json.dumps(asdict(baseline), ensure_ascii=False) + "\n")

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
                            result.extra["translation_step"] = asdict(history[-1])
                            sink.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    build_reports(results_path, output_dir)
    return results_path


def _resolve(config_path: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path
