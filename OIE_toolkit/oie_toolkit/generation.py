"""Local (GPU) LLM helpers built on top of Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    SUPPORTED_FORMATS,
    build_conversion_prompt,
)


@dataclass(slots=True)
class LocalGenerationConfig:
    """Runtime configuration for :class:`LocalLLMExtractor`."""

    model_name: str = "microsoft/Phi-3-mini-128k-instruct"
    temperature: float | None = None
    top_p: float | None = None
    max_new_tokens: int = 8192
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    schema_description: str | None = None
    extra_requirements: str | None = None
    device_map: str | None = "auto"
    torch_dtype: str | None = "bfloat16"
    trust_remote_code: bool = False
    use_chat_template: bool = True


class LocalLLMExtractor:
    """Convert notes with locally-hosted models (Phi-3.5, Qwen2.5, Llama, etc.)."""

    def __init__(
        self,
        *,
        config: LocalGenerationConfig | None = None,
        text_generation_pipeline: Optional[Any] = None,
    ) -> None:
        self.config = config or LocalGenerationConfig()
        self.generator = (
            text_generation_pipeline
            if text_generation_pipeline is not None
            else self._build_pipeline()
        )
        self.tokenizer = self.generator.tokenizer  # type: ignore[attr-defined]

    def _resolve_dtype(self) -> torch.dtype | None:
        if not self.config.torch_dtype:
            return None
        if not hasattr(torch, self.config.torch_dtype):
            raise ValueError(f"Unknown torch dtype: {self.config.torch_dtype}")
        return getattr(torch, self.config.torch_dtype)

    def _build_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model_kwargs: Dict[str, object] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        dtype = self._resolve_dtype()
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        pipeline_kwargs: Dict[str, object] = {
            "model": model,
            "tokenizer": tokenizer,
            "top_k": None,
        }
        if dtype is not None:
            pipeline_kwargs["torch_dtype"] = dtype
        text_gen = pipeline("text-generation", **pipeline_kwargs)
        return text_gen

    def _build_prompt(self, note_text: str, *, note_id: str, fmt: str) -> str:
        prompt = build_conversion_prompt(
            note_text,
            note_id=note_id,
            output_format=fmt,
            schema_description=self.config.schema_description,
            extra_requirements=self.config.extra_requirements,
        )
        if (
            self.config.use_chat_template
            and hasattr(self.tokenizer, "apply_chat_template")
        ):
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self.config.system_prompt}\n\n{prompt}"

    def convert_note(self, note_text: str, *, note_id: str, fmt: str) -> str:
        fmt_normalized = fmt.lower()
        if fmt_normalized not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Choose from {SUPPORTED_FORMATS}.")

        prepared_prompt = self._build_prompt(
            note_text,
            note_id=note_id,
            fmt=fmt_normalized,
        )
        generation_kwargs: Dict[str, object] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }
        if self.config.temperature is not None:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["do_sample"] = self.config.temperature > 0
        if self.config.top_p is not None:
            generation_kwargs["top_p"] = self.config.top_p
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        outputs = self.generator(
            prepared_prompt,
            **generation_kwargs,
        )
        generated_text = outputs[0]["generated_text"]
        if generated_text.startswith(prepared_prompt):
            response = generated_text[len(prepared_prompt) :].strip()
        else:
            response = generated_text.strip()
        return response


__all__ = ["LocalGenerationConfig", "LocalLLMExtractor"]
