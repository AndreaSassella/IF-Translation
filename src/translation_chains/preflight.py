from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List

from .datasets import load_dataset_from_config


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

    if config.get("translation_engine") == "nllb":
        supported = {"en", "fr", "de", "es", "ru", "ar", "hi", "ja"}
        for path in config.get("paths", []):
            bad = [code for code in path if code not in supported]
            if bad:
                findings.append({"level": "error", "message": f"Unsupported NLLB language code(s): {bad}"})

    status = "pass" if not any(item["level"] == "error" for item in findings) else "fail"
    return {"status": status, "findings": findings}
