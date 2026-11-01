"""Deterministic adapter for tests, with injectable failures."""

from __future__ import annotations

from typing import Any

from src.models.base import CaptionAdapter, CaptionRequest, CaptionResult, TransientError, classify


class MockAdapter(CaptionAdapter):
    """`model_config["mock"]` maps sample_id -> behaviour:

    - "transient:N": raise TransientError on the first N calls, then succeed
    - "error": raise a permanent error
    - "empty" / "truncated": return that status
    """

    def __init__(self, model_key: str, model_config: dict[str, Any], generation_config: dict[str, Any]):
        super().__init__(model_key, model_config, generation_config)
        self.behaviour = dict(model_config.get("mock") or {})
        self.calls: dict[str, int] = {}

    def caption(self, request: CaptionRequest) -> CaptionResult:
        self.calls[request.sample_id] = self.calls.get(request.sample_id, 0) + 1
        behaviour = self.behaviour.get(request.sample_id, "ok")
        if behaviour.startswith("transient:") and self.calls[request.sample_id] <= int(behaviour.split(":")[1]):
            raise TransientError(f"simulated transient failure {self.calls[request.sample_id]}")
        if behaviour == "error":
            raise ValueError("simulated permanent failure")
        caption = "" if behaviour == "empty" else f"Mock caption for {request.sample_id}."
        return CaptionResult(
            sample_id=request.sample_id,
            status=classify(caption, truncated=behaviour == "truncated"),
            caption=caption,
            model_id=self.model_id,
            model_version="mock-1",
            raw_response={"text": caption},
            request_params=dict(request.decoding),
            finish_reason="length" if behaviour == "truncated" else "stop",
        )
