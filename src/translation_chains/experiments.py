from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from .adapters import make_model_adapter, make_translation_adapter
from .datasets import load_dataset_from_config
from .evaluators import evaluate_record
from .reporting import build_reports
from .schemas import ExperimentStatus, TranslationStep
from .status import write_status

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def run_experiment(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    output_dir = _resolve(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_from_config(config_path, config)
    translation = make_translation_adapter(
        config["translation_engine"],
        config.get("translation_adapter", {}),
    )
    models = [_make_model_from_spec(spec) for spec in config["models"]]
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

    total_steps_per_record = 1 + (len(config["regimes"]) * sum(max(len(path) - 1, 0) for path in config["paths"]))
    overall_total = len(models) * len(dataset) * total_steps_per_record
    eta_warmup_rows = int(config.get("eta_warmup_rows", 50))
    overall_bar = _make_progress_bar(
        total=overall_total,
        desc="Overall experiment progress",
        position=0,
    )
    start_time = perf_counter()
    translation_cache: Dict[tuple, str] = {}
    translation_histories: Dict[tuple, List[TranslationStep]] = {}
    eta_reported = False

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as sink:
        for model_index, model in enumerate(models, start=1):
            model_bar = _make_progress_bar(
                total=len(dataset) * total_steps_per_record,
                desc=f"Model {model_index}/{len(models)}: {model.name}",
                position=1,
            )
            for record in dataset:
                if not eta_reported and status.completed_rows >= max(1, eta_warmup_rows * total_steps_per_record):
                    _record_eta_note(status, status_path, start_time, overall_total)
                    eta_reported = True

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
                _progress_update(overall_bar, 1)
                _progress_update(model_bar, 1)

                for regime in config["regimes"]:
                    for path in config["paths"]:
                        history = _get_or_build_translation_history(
                            translation=translation,
                            translation_cache=translation_cache,
                            history_cache=translation_histories,
                            prompt_id=record.prompt_id,
                            original_prompt=record.original_prompt,
                            path=path,
                        )
                        for depth, (source_lang, target_lang) in enumerate(
                            zip(path, path[1:]),
                            start=1,
                        ):
                            step = history[depth - 1]
                            translated_prompt = step.output_text

                            evaluated_prompt = translated_prompt
                            if regime == "back_translated" and target_lang != "en":
                                evaluated_prompt = _translate_cached(
                                    translation=translation,
                                    cache=translation_cache,
                                    prompt_id=record.prompt_id,
                                    text=translated_prompt,
                                    source_lang=target_lang,
                                    target_lang="en",
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
                            result.extra["translation_step"] = asdict(step)
                            sink.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                            status.completed_rows += 1
                            status.updated_at = datetime.now(timezone.utc).isoformat()
                            write_status(status_path, status)
                            _progress_update(overall_bar, 1)
                            _progress_update(model_bar, 1)
            _progress_close(model_bar)

    status.status = "completed"
    status.finished_at = datetime.now(timezone.utc).isoformat()
    status.updated_at = status.finished_at
    write_status(status_path, status)
    _progress_close(overall_bar)
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


def _make_model_from_spec(spec: Any):
    if isinstance(spec, str):
        return make_model_adapter(spec, {})
    if isinstance(spec, dict):
        adapter_type = spec.get("type")
        if adapter_type == "huggingface_generation":
            return make_model_adapter(spec["model_id"], spec)
        return make_model_adapter(spec["name"], spec)
    raise ValueError(f"Unsupported model spec: {spec}")


def _get_or_build_translation_history(
    translation,
    translation_cache: Dict[tuple, str],
    history_cache: Dict[tuple, List[TranslationStep]],
    prompt_id: str,
    original_prompt: str,
    path: List[str],
) -> List[TranslationStep]:
    history_key = (prompt_id, tuple(path), translation.name)
    if history_key in history_cache:
        return history_cache[history_key]

    translated_prompt = original_prompt
    history: List[TranslationStep] = []
    for depth, (source_lang, target_lang) in enumerate(zip(path, path[1:]), start=1):
        translated_prompt = _translate_cached(
            translation=translation,
            cache=translation_cache,
            prompt_id=prompt_id,
            text=translated_prompt,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        history.append(
            TranslationStep(
                step_index=depth,
                source_lang=source_lang,
                target_lang=target_lang,
                input_text=original_prompt if depth == 1 else history[-1].output_text,
                output_text=translated_prompt,
                engine_name=translation.name,
            )
        )
    history_cache[history_key] = history
    return history


def _translate_cached(
    translation,
    cache: Dict[tuple, str],
    prompt_id: str,
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    key = (prompt_id, source_lang, target_lang, text, translation.name)
    if key not in cache:
        cache[key] = translation.translate(text, source_lang, target_lang)
    return cache[key]


def _record_eta_note(
    status: ExperimentStatus,
    status_path: Path,
    start_time: float,
    overall_total: int,
) -> None:
    elapsed = max(perf_counter() - start_time, 1e-9)
    rate = status.completed_rows / elapsed if status.completed_rows else 0.0
    remaining = max(overall_total - status.completed_rows, 0)
    remaining_seconds = remaining / rate if rate > 0 else 0.0
    hours = remaining_seconds / 3600.0
    note = (
        f"ETA estimate after warmup: {rate:.2f} rows/sec, "
        f"approximately {hours:.2f} hours remaining."
    )
    status.notes.append(note)
    write_status(status_path, status)


def _make_progress_bar(total: int, desc: str, position: int):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, position=position, leave=True)


def _progress_update(bar, amount: int) -> None:
    if bar is not None:
        bar.update(amount)


def _progress_close(bar) -> None:
    if bar is not None:
        bar.close()
