"""Audio Flamingo 3 (transformers `-hf` checkpoint)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from src.models.audio import load_mono_float
from src.models.base import CaptionRequest, CaptionResult, classify
from src.models.hf_common import TARGET_SAMPLE_RATE, HFAdapterBase


class AudioFlamingo3Adapter(HFAdapterBase):
    def load(self) -> None:
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=self.dtype, device_map="auto"
        )
        self.workdir = Path(tempfile.mkdtemp(prefix="af3_"))

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system_prompt": None, "repetition_penalty": None}

    def caption(self, request: CaptionRequest) -> CaptionResult:
        # The documented chat-template input is a file path, so the soxr-resampled 16 kHz audio is written to a
        # temporary float WAV; the processor then loads it at its native rate without resampling again.
        audio, rate, preprocessing = load_mono_float(request.audio_path, TARGET_SAMPLE_RATE)
        resampled = self.workdir / f"{request.sample_id}.wav"
        sf.write(resampled, audio, rate, subtype="FLOAT")
        preprocessing["model_input"] = "temporary 16 kHz float32 WAV passed by path"
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": request.prompt}, {"type": "audio", "path": str(resampled)}],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_dict=True
            ).to(self.model.device)
            kwargs, deviations = self.generate_kwargs(request.decoding)
            with self.torch.inference_mode():
                sequences = self.model.generate(**inputs, **kwargs)
        finally:
            resampled.unlink(missing_ok=True)

        new_tokens = sequences[0, inputs["input_ids"].shape[1] :]
        caption = self.processor.batch_decode(new_tokens.unsqueeze(0), skip_special_tokens=True)[0].strip()
        truncated = self.is_truncated(new_tokens, kwargs["max_new_tokens"], self.eos_ids(self.processor.tokenizer))
        return CaptionResult(
            sample_id=request.sample_id,
            status=classify(caption, truncated),
            caption=caption,
            model_id=self.model_id,
            model_version=self.revision,
            raw_response={"text": caption, "new_token_ids": new_tokens.tolist()},
            request_params=kwargs,
            deviations=deviations,
            audio_preprocessing=preprocessing,
            usage={
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": len(new_tokens),
                "peak_gpu_memory_gb": self.peak_memory_gb(),
            },
            finish_reason="length" if truncated else "stop",
        )
