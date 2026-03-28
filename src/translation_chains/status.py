from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .schemas import ExperimentStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, status: ExperimentStatus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(status), handle, indent=2)


def read_status(path: Path) -> ExperimentStatus:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentStatus(**payload)
