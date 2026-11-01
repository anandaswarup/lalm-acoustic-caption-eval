"""GPT-Audio-1.5 through OpenAI Chat Completions (audio in, text out)."""

from __future__ import annotations

import base64
import os
from typing import Any

from src.models.audio import read_wav_bytes
from src.models.base import CaptionAdapter, CaptionRequest, CaptionResult, TransientError, classify

# generation.yaml decoding key -> Chat Completions parameter
PARAMETERS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "max_output_tokens": "max_completion_tokens",
}


def _rejected_parameter(error: Exception, params: dict[str, Any]) -> str | None:
    """Name of the sampling parameter an API error rejects, if any."""
    param = getattr(error, "param", None)
    if param in params:
        return param
    message = str(error).lower()
    for name in params:
        if name in PARAMETERS.values() and f"'{name}'" in message and "unsupported" in message:
            return name
    return None


class OpenAIAudioAdapter(CaptionAdapter):
    def setup(self) -> None:
        import openai

        self.openai = openai
        self.client = openai.OpenAI(
            api_key=os.environ[self.model_config["api_key_env"]], timeout=self.timeout, max_retries=0
        )
        self.dropped: dict[str, str] = {}  # parameters the API rejected, kept dropped for the rest of the run

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "library": "openai", "library_version": self.openai.__version__}

    def build_params(self, request: CaptionRequest) -> dict[str, Any]:
        params = {api: request.decoding[key] for key, api in PARAMETERS.items() if key in request.decoding}
        return {name: value for name, value in params.items() if name not in self.dropped}

    def caption(self, request: CaptionRequest) -> CaptionResult:
        audio, preprocessing = read_wav_bytes(request.audio_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": "wav"}},
                ],
            }
        ]
        params = self.build_params(request)
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id, messages=messages, modalities=["text"], store=False, **params
                )
                break
            except (
                self.openai.APITimeoutError,
                self.openai.APIConnectionError,
                self.openai.RateLimitError,
                self.openai.InternalServerError,
            ) as error:
                raise TransientError(str(error)) from error
            except self.openai.BadRequestError as error:
                rejected = _rejected_parameter(error, params)
                if rejected is None:
                    raise
                self.dropped[rejected] = str(error)
                params.pop(rejected)

        choice = response.choices[0]
        caption = (choice.message.content or "").strip()
        usage = response.usage.model_dump() if response.usage else {}
        return CaptionResult(
            sample_id=request.sample_id,
            status=classify(caption, truncated=choice.finish_reason == "length"),
            caption=caption,
            model_id=self.model_id,
            model_version=response.model,
            raw_response=response.model_dump(mode="json"),
            request_params={"modalities": ["text"], **params},
            deviations=[f"parameter '{name}' rejected by API, not sent" for name in self.dropped],
            audio_preprocessing=preprocessing,
            usage={**usage, "system_fingerprint": response.system_fingerprint},
            finish_reason=choice.finish_reason,
        )
