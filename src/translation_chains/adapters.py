from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List

from .schemas import PromptRecord


class TranslationAdapter:
    name = "base"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError


class IdentityTranslationAdapter(TranslationAdapter):
    name = "identity"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return text


class DriftTranslationAdapter(TranslationAdapter):
    """
    Deterministic local adapter that simulates translation drift.

    This is not a real MT system. It exists so the repository is executable
    without network access or third-party dependencies.
    """

    name = "drift"

    _LANG_REPLACEMENTS: Dict[str, Dict[str, str]] = {
        "fr": {"exactly": "approximately", "valid": "well-formed", "short": "brief"},
        "de": {"include": "mention", "exactly": "precisely", "comma-separated": "listed"},
        "ja": {"bullet points": "items", "paragraph": "text block", "JSON": "structured data"},
        "ar": {"do not use": "avoid", "title": "heading", "sentences": "lines"},
        "ru": {"respond": "answer", "keys": "fields", "colors": "colours"},
        "hi": {"about": "regarding", "write": "produce", "starts with": "begins with"},
    }

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        seed = self._stable_seed(text, source_lang, target_lang)
        rng = random.Random(seed)
        out = text

        for source, target in self._LANG_REPLACEMENTS.get(target_lang, {}).items():
            out = re.sub(rf"\b{re.escape(source)}\b", target, out, flags=re.IGNORECASE)

        if rng.random() < 0.35:
            out = out.replace("exactly", "about")
        if rng.random() < 0.30:
            out = out.replace("Do not use", "Try not to use")
        if rng.random() < 0.20:
            out = out.replace("valid JSON", "JSON")
        if rng.random() < 0.25:
            out = out.replace("comma-separated", "separated")

        return f"[{target_lang}] {out}"

    @staticmethod
    def _stable_seed(text: str, source_lang: str, target_lang: str) -> int:
        token = f"{source_lang}>{target_lang}:{text}"
        return sum(ord(char) for char in token)


class ModelAdapter:
    name = "base"

    def generate(self, record: PromptRecord, prompt_text: str) -> str:
        raise NotImplementedError


class EchoModelAdapter(ModelAdapter):
    name = "echo"

    def generate(self, record: PromptRecord, prompt_text: str) -> str:
        return prompt_text


class ReferenceModelAdapter(ModelAdapter):
    name = "reference"

    def generate(self, record: PromptRecord, prompt_text: str) -> str:
        return record.ideal_response or ""


class FragileModelAdapter(ModelAdapter):
    name = "fragile"

    def generate(self, record: PromptRecord, prompt_text: str) -> str:
        drift_score = prompt_text.lower().count("[")
        response = record.ideal_response or ""

        if "approximately" in prompt_text.lower() or "about" in prompt_text.lower():
            response = response.replace("- ", "", 1)

        if "structured data" in prompt_text.lower():
            response = "topic: volcanoes; risk: ash and lava"

        if "items" in prompt_text.lower():
            response = response.replace("- ", "", 1)

        if "try not to use" in prompt_text.lower() and "Earth" not in response:
            response = f"{response}\nEarth"

        if "separated" in prompt_text.lower() and "comma-separated" not in prompt_text.lower():
            response = response.replace(", ", " | ")

        if drift_score >= 3 and record.prompt_id == "p2":
            response = "Rainforests support biodiversity and store carbon."
        if drift_score >= 3 and record.prompt_id == "p3":
            response = "Climate Futures\nClimate change affects ecosystems and people across the world in complex and uneven ways."

        return response


def make_translation_adapter(name: str) -> TranslationAdapter:
    registry = {
        "identity": IdentityTranslationAdapter,
        "drift": DriftTranslationAdapter,
    }
    if name not in registry:
        raise ValueError(f"Unknown translation adapter: {name}")
    return registry[name]()


def make_model_adapter(name: str) -> ModelAdapter:
    registry = {
        "echo": EchoModelAdapter,
        "reference": ReferenceModelAdapter,
        "fragile": FragileModelAdapter,
    }
    if name not in registry:
        raise ValueError(f"Unknown model adapter: {name}")
    return registry[name]()


@dataclass
class AdapterSpec:
    name: str
    kind: str

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "kind": self.kind}, indent=2)
