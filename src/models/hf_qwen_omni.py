"""Qwen2.5-Omni-7B (transformers), text output only."""

from __future__ import annotations

from typing import Any

from src.models.audio import load_mono_float
from src.models.base import CaptionRequest, CaptionResult, classify
from src.models.hf_common import TARGET_SAMPLE_RATE, HFAdapterBase


class Qwen25OmniAdapter(HFAdapterBase):
    def load(self) -> None:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=self.dtype, device_map="auto", attn_implementation="sdpa"
        )
        self.model.disable_talker()  # text only; saves ~2 GB

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "attn_implementation": "sdpa", "talker": "disabled", "system_prompt": None}

    def caption(self, request: CaptionRequest) -> CaptionResult:
        audio, rate, preprocessing = load_mono_float(request.audio_path, TARGET_SAMPLE_RATE)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": request.audio_path.name},
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[audio], sampling_rate=rate, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.dtype)
        kwargs, deviations = self.generate_kwargs(request.decoding)

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, return_audio=False, **kwargs)
        sequences = output[0] if isinstance(output, tuple) else output
        new_tokens = sequences[0, inputs["input_ids"].shape[1] :]
        caption = self.processor.batch_decode(
            new_tokens.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        truncated = self.is_truncated(new_tokens, kwargs["max_new_tokens"], self.eos_ids(self.processor.tokenizer))
        return CaptionResult(
            sample_id=request.sample_id,
            status=classify(caption, truncated),
            caption=caption,
            model_id=self.model_id,
            model_version=self.revision,
            raw_response={"text": caption, "prompt_text": text, "new_token_ids": new_tokens.tolist()},
            request_params={**kwargs, "return_audio": False},
            deviations=deviations,
            audio_preprocessing=preprocessing,
            usage={
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": len(new_tokens),
                "peak_gpu_memory_gb": self.peak_memory_gb(),
            },
            finish_reason="length" if truncated else "stop",
        )
