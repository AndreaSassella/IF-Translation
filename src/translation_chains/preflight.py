from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List

from .datasets import load_dataset_from_config
from .nltk_setup import ensure_ifeval_nltk_resources


def run_preflight(config_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    findings: List[Dict[str, str]] = []

    try:
        dataset = load_dataset_from_config(config_path, config)
        findings.append({"level": "info", "message": f"Loaded dataset with {len(dataset)} records."})
    except Exception as exc:
        findings.append({"level": "error", "message": f"Dataset loading failed: {exc}"})

    package_requirements = {
        "datasets": "datasets",
        "transformers": "transformers",
        "torch": "torch",
        "instruction_following_eval": "instruction_following_eval",
    }
    for import_name, label in package_requirements.items():
        try:
            importlib.import_module(import_name)
            findings.append({"level": "info", "message": f"Package available: {label}"})
        except Exception as exc:
            findings.append({"level": "warning", "message": f"Package missing or broken: {label} ({exc})"})

    try:
        import torch

        cuda_version = getattr(torch.version, "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        findings.append(
            {
                "level": "info",
                "message": (
                    f"PyTorch CUDA check: torch={torch.__version__}, "
                    f"torch_cuda={cuda_version}, cuda_available={cuda_available}, "
                    f"device_count={device_count}"
                ),
            }
        )
        if device_count > 0 and not cuda_available:
            findings.append(
                {
                    "level": "warning",
                    "message": (
                        "CUDA devices are visible but torch.cuda.is_available() is False. "
                        "This usually means a driver / CUDA runtime / PyTorch build mismatch."
                    ),
                }
            )
    except Exception as exc:
        findings.append({"level": "warning", "message": f"PyTorch CUDA preflight failed: {exc}"})

    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("name") in {"google/IFEval", "HuggingFaceH4/ifeval", "IFEval"}:
        try:
            ensure_ifeval_nltk_resources()
            findings.append({"level": "info", "message": "NLTK IFEval tokenizer resources are available."})
        except Exception as exc:
            findings.append({"level": "warning", "message": f"NLTK IFEval resource setup failed: {exc}"})

    if config.get("translation_engine") == "nllb":
        supported = {"en", "fr", "de", "es", "ru", "ar", "hi", "ja"}
        for path in config.get("paths", []):
            bad = [code for code in path if code not in supported]
            if bad:
                findings.append({"level": "error", "message": f"Unsupported NLLB language code(s): {bad}"})

    status = "pass" if not any(item["level"] == "error" for item in findings) else "fail"
    return {"status": status, "findings": findings}
