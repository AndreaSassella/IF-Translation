from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


class HuggingFaceGenerationAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device_map: str = "auto",
        torch_dtype: Optional[str] = None,
    ) -> None:
        self.name = model_id
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._pipeline = None

    def generate(self, record: PromptRecord, prompt_text: str) -> str:
        pipe = self._get_pipeline()
        try:
            output = pipe([{"role": "user", "content": prompt_text}], return_full_text=False)
        except Exception:
            output = pipe(prompt_text, return_full_text=False)
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                if "generated_text" in first:
                    generated = first["generated_text"]
                    if isinstance(generated, list) and generated:
                        last = generated[-1]
                        if isinstance(last, dict):
                            return str(last.get("content", ""))
                    return str(generated)
        return str(output)

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "Hugging Face model adapters require `transformers` and `torch`. "
                "Install requirements.txt before using adapter.type='huggingface_generation'."
            ) from exc

        dtype = None
        if self.torch_dtype:
            dtype = getattr(torch, self.torch_dtype)

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            device_map=self.device_map,
            torch_dtype=dtype,
        )
        if hasattr(self._pipeline, "model") and hasattr(self._pipeline.model, "generation_config"):
            generation_config = self._pipeline.model.generation_config
            if hasattr(generation_config, "max_new_tokens"):
                generation_config.max_new_tokens = self.max_new_tokens
            if hasattr(generation_config, "max_length"):
                generation_config.max_length = None
            if hasattr(generation_config, "do_sample"):
                generation_config.do_sample = self.temperature > 0
            if self.temperature > 0:
                if hasattr(generation_config, "temperature"):
                    generation_config.temperature = self.temperature
                if hasattr(generation_config, "top_p"):
                    generation_config.top_p = self.top_p
            else:
                if hasattr(generation_config, "temperature"):
                    generation_config.temperature = None
                if hasattr(generation_config, "top_p"):
                    generation_config.top_p = None
                if hasattr(generation_config, "top_k"):
                    generation_config.top_k = None
        device = getattr(self._pipeline.model, "device", "unknown")
        hf_map = getattr(self._pipeline.model, "hf_device_map", None)
        print(f"[LLM] {self.model_id} loaded on device={device}, hf_device_map={hf_map}")
        return self._pipeline


class NLLBTranslationAdapter(TranslationAdapter):
    name = "nllb"

    _LANG_MAP = {
        "en": "eng_Latn",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "es": "spa_Latn",
        "ru": "rus_Cyrl",
        "ar": "arb_Arab",
        "hi": "hin_Deva",
        "ja": "jpn_Jpan",
    }

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        device_map: str = "auto",
        max_length: int = 1024,
        torch_dtype: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.max_length = max_length
        self.torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        tokenizer, model = self._get_components()
        src = self._LANG_MAP[source_lang]
        tgt = self._LANG_MAP[target_lang]
        tokenizer.src_lang = src
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        if hasattr(model, "device"):
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),
            max_length=self.max_length,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def _get_components(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The NLLB translation adapter requires `transformers` and `torch`. "
                "Install requirements.txt before using translation_engine='nllb'."
            ) from exc
        dtype = None
        if self.torch_dtype:
            dtype = getattr(torch, self.torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=dtype,
        )
        device = getattr(self._model, "device", "unknown")
        hf_map = getattr(self._model, "hf_device_map", None)
        print(f"[MT] {self.model_id} loaded on device={device}, hf_device_map={hf_map}")
        return self._tokenizer, self._model


def make_translation_adapter(name: str, config: Optional[Dict[str, Any]] = None) -> TranslationAdapter:
    config = config or {}
    registry = {
        "identity": IdentityTranslationAdapter,
        "drift": DriftTranslationAdapter,
    }
    if name == "nllb":
        return NLLBTranslationAdapter(
            model_id=config.get("model_id", "facebook/nllb-200-distilled-600M"),
            device_map=config.get("device_map", "auto"),
            max_length=config.get("max_length", 1024),
            torch_dtype=config.get("torch_dtype"),
        )
    if name not in registry:
        raise ValueError(f"Unknown translation adapter: {name}")
    return registry[name]()


def make_model_adapter(name: str, config: Optional[Dict[str, Any]] = None) -> ModelAdapter:
    config = config or {}
    registry = {
        "echo": EchoModelAdapter,
        "reference": ReferenceModelAdapter,
        "fragile": FragileModelAdapter,
    }
    if config.get("type") == "huggingface_generation":
        return HuggingFaceGenerationAdapter(
            model_id=config["model_id"],
            max_new_tokens=config.get("max_new_tokens", 512),
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
            device_map=config.get("device_map", "auto"),
            torch_dtype=config.get("torch_dtype"),
        )
    if name not in registry:
        raise ValueError(f"Unknown model adapter: {name}")
    return registry[name]()


@dataclass
class AdapterSpec:
    name: str
    kind: str

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "kind": self.kind}, indent=2)
