"""Gemini 3.8 Flash through google-genai (inline WAV in, text out)."""

from __future__ import annotations

import os
from typing import Any

from src.models.audio import read_wav_bytes
from src.models.base import CaptionAdapter, CaptionRequest, CaptionResult, TransientError, classify

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}


class GeminiAdapter(CaptionAdapter):
    def setup(self) -> None:
        from google import genai
        from google.genai import errors, types

        self.genai, self.errors, self.types = genai, errors, types
        self.client = genai.Client(
            api_key=os.environ[self.model_config["api_key_env"]],
            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "library": "google-genai", "library_version": self.genai.__version__}

    def build_config(self, decoding: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        keys = ("temperature", "top_p", "seed", "max_output_tokens")
        params = {key: decoding[key] for key in keys if key in decoding}
        if "thinking_level" in decoding:
            params["thinking_level"] = decoding["thinking_level"]
        thinking = (
            self.types.ThinkingConfig(thinking_level=params["thinking_level"]) if "thinking_level" in params else None
        )
        config = self.types.GenerateContentConfig(
            **{key: value for key, value in params.items() if key != "thinking_level"}, thinking_config=thinking
        )
        return config, params

    def caption(self, request: CaptionRequest) -> CaptionResult:
        audio, preprocessing = read_wav_bytes(request.audio_path)
        config, params = self.build_config(request.decoding)
        contents = [request.prompt, self.types.Part.from_bytes(data=audio, mime_type="audio/wav")]
        try:
            response = self.client.models.generate_content(model=self.model_id, contents=contents, config=config)
        except self.errors.APIError as error:
            if getattr(error, "code", None) in TRANSIENT_CODES:
                raise TransientError(str(error)) from error
            raise

        candidate = response.candidates[0] if response.candidates else None
        finish_reason = getattr(candidate.finish_reason, "name", str(candidate.finish_reason)) if candidate else None
        caption = (response.text or "").strip()
        usage = response.usage_metadata.model_dump(mode="json", exclude_none=True) if response.usage_metadata else {}
        return CaptionResult(
            sample_id=request.sample_id,
            status=classify(caption, truncated=finish_reason == "MAX_TOKENS"),
            caption=caption,
            model_id=self.model_id,
            model_version=response.model_version,
            raw_response=response.model_dump(mode="json", exclude_none=True),
            request_params=params,
            deviations=["max_output_tokens includes thinking tokens"] if "thinking_level" in params else [],
            audio_preprocessing=preprocessing,
            usage=usage,
            finish_reason=finish_reason,
        )
