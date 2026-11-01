"""Audio Flamingo 3 (transformers `-hf` checkpoint)."""

from __future__ import annotations

from typing import Any

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

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system_prompt": None, "repetition_penalty": None}

    def caption(self, request: CaptionRequest) -> CaptionResult:
        # The soxr-resampled 16 kHz array goes into the chat template as `audio`; transformers passes arrays through
        # unchanged (a `path` would be decoded and resampled again, and needs librosa).
        audio, rate, preprocessing = load_mono_float(request.audio_path, TARGET_SAMPLE_RATE)
        preprocessing["model_input"] = "16 kHz float32 array in the chat template"
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": request.prompt}, {"type": "audio", "audio": audio}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True, sampling_rate=rate
        ).to(self.model.device, dtype=self.dtype)  # casts the float32 audio features to bf16; ids stay integer
        kwargs, deviations = self.generate_kwargs(request.decoding)
        with self.torch.inference_mode():
            sequences = self.model.generate(**inputs, **kwargs)

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
