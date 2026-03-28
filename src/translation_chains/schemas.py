from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Constraint:
    type: str
    value: Any
    category: str = "uncategorized"


@dataclass
class PromptRecord:
    prompt_id: str
    original_prompt: str
    constraints: List[Constraint]
    ideal_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentStatus:
    run_name: str
    status: str
    dataset_source: str
    dataset_size: int
    expected_rows: int
    completed_rows: int = 0
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    finished_at: Optional[str] = None
    output_dir: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class TranslationStep:
    step_index: int
    source_lang: str
    target_lang: str
    input_text: str
    output_text: str
    engine_name: str
    quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    model_name: str
    prompt_id: str
    regime: str
    language_path: List[str]
    depth: int
    evaluated_prompt: str
    raw_response: str
    prompt_level_strict: Optional[float] = None
    prompt_level_loose: Optional[float] = None
    instruction_level_strict: Optional[float] = None
    instruction_level_loose: Optional[float] = None
    category_scores: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
