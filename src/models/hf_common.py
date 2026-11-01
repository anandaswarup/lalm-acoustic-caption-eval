"""Shared pieces of the open-weight (transformers) adapters; these run on Hugging Face Jobs."""

from __future__ import annotations

from typing import Any

from src.models.base import CaptionAdapter

TARGET_SAMPLE_RATE = 16000  # Whisper-style feature extractors of both open-weight models


class HFAdapterBase(CaptionAdapter):
    """Loads torch lazily, maps the shared decoding settings to `generate`, and reports hardware."""

    def setup(self) -> None:
        import torch
        import transformers

        self.torch, self.transformers = torch, transformers
        self.revision = self.model_config.get("revision")
        if not self.revision:
            raise ValueError(f"Model '{self.model_key}' needs a pinned revision")
        self.dtype = torch.bfloat16
        self.load()
        self.model.eval()

    def load(self) -> None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        torch = self.torch
        details = {
            **super().describe(),
            "revision": self.revision,
            "library": "transformers",
            "library_version": self.transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": str(self.dtype),
            "cuda": torch.version.cuda,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            details["gpu"] = {
                "name": props.name,
                "memory_gb": round(props.total_memory / 1e9, 1),
                "count": torch.cuda.device_count(),
            }
        return details

    def generate_kwargs(self, decoding: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Greedy decoding for temperature 0; anything else is sampled with the given temperature/top_p."""
        kwargs: dict[str, Any] = {"max_new_tokens": decoding["max_output_tokens"]}
        deviations = []
        if decoding.get("temperature", 0.0) == 0.0:
            kwargs["do_sample"] = False
            if decoding.get("top_p", 1.0) != 1.0:
                deviations.append("top_p ignored under greedy decoding")
        else:
            kwargs.update(do_sample=True, temperature=decoding["temperature"], top_p=decoding.get("top_p", 1.0))
        return kwargs, deviations

    def is_truncated(self, new_tokens: Any, max_new_tokens: int, eos_token_ids: set[int]) -> bool:
        tokens = new_tokens.tolist()
        return len(tokens) >= max_new_tokens and (not tokens or tokens[-1] not in eos_token_ids)

    def peak_memory_gb(self) -> float | None:
        torch = self.torch
        return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None

    def eos_ids(self, tokenizer: Any) -> set[int]:
        ids = self.model.generation_config.eos_token_id
        ids = set(ids if isinstance(ids, list) else [ids]) if ids is not None else set()
        if tokenizer.eos_token_id is not None:
            ids.add(tokenizer.eos_token_id)
        return ids
