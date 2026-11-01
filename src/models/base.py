"""Shared caption-generation interface for all LALMs (E04)."""

from __future__ import annotations

import importlib
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("ok", "empty", "truncated", "error")

# Adapter name (models.yaml `adapter`) -> "module:class"; imported lazily so that
# SDKs and torch are only needed for the adapter actually used.
ADAPTERS = {
    "mock": "src.models.mock:MockAdapter",
    "openai_audio": "src.models.openai_audio:OpenAIAudioAdapter",
    "gemini": "src.models.gemini:GeminiAdapter",
    "qwen2_5_omni": "src.models.hf_qwen_omni:Qwen25OmniAdapter",
    "audio_flamingo_3": "src.models.hf_audio_flamingo:AudioFlamingo3Adapter",
}


class TransientError(Exception):
    """A failure worth retrying: timeouts, rate limits, server errors."""


@dataclass
class CaptionRequest:
    sample_id: str
    audio_path: Path
    prompt: str
    decoding: dict[str, Any]


@dataclass
class CaptionResult:
    sample_id: str
    status: str
    caption: str | None
    model_id: str
    model_version: str | None = None
    raw_response: Any = None
    request_params: dict[str, Any] = field(default_factory=dict)
    deviations: list[str] = field(default_factory=list)
    audio_preprocessing: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None
    error_traceback: str | None = None  # last frames, for exceptions only
    # Filled in by the runner.
    model_key: str | None = None
    prompt_sha256: str | None = None
    latency_s: float | None = None
    attempts: int | None = None
    timestamp_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(caption: str | None, truncated: bool) -> str:
    """Status of a completed request: truncated beats empty, which beats ok."""
    if truncated:
        return "truncated"
    if caption is None or not caption.strip():
        return "empty"
    return "ok"


def merge_decoding(generation_config: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    return {**generation_config.get("decoding", {}), **(model_config.get("decoding_overrides") or {})}


class CaptionAdapter(ABC):
    """One audio recording and a prompt in, one caption out.

    Raise `TransientError` for failures the runner should retry; any other exception is recorded as an error.
    """

    def __init__(self, model_key: str, model_config: dict[str, Any], generation_config: dict[str, Any]):
        self.model_key = model_key
        self.model_config = model_config
        self.generation_config = generation_config
        self.model_id = model_config["model_id"]
        self.timeout = generation_config.get("requests", {}).get("timeout_seconds", 120)

    def setup(self) -> None:
        """Create clients or load weights; called once before the first request."""
        return None

    @abstractmethod
    def caption(self, request: CaptionRequest) -> CaptionResult: ...

    def describe(self) -> dict[str, Any]:
        """Adapter details for the run metadata (library versions, hardware, revision)."""
        return {"adapter": type(self).__name__, "model_id": self.model_id}

    def error_result(self, request: CaptionRequest, error: Exception | str) -> CaptionResult:
        is_exception = isinstance(error, Exception)
        return CaptionResult(
            sample_id=request.sample_id,
            status="error",
            caption=None,
            model_id=self.model_id,
            error=f"{type(error).__name__}: {error}" if is_exception else error,
            error_traceback="".join(traceback.format_exception(error, limit=-8)) if is_exception else None,
        )


def load_adapter(model_key: str, model_config: dict[str, Any], generation_config: dict[str, Any]) -> CaptionAdapter:
    name = model_config.get("adapter")
    if name not in ADAPTERS:
        raise ValueError(f"Model '{model_key}' has unknown adapter {name!r}; expected one of {sorted(ADAPTERS)}")
    module_name, class_name = ADAPTERS[name].split(":")
    adapter_class = getattr(importlib.import_module(module_name), class_name)
    return adapter_class(model_key, model_config, generation_config)
