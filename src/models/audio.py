"""Audio loading and provider-specific preprocessing. Audio is never cropped."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def read_wav_bytes(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    """Original file bytes, sent unchanged to API providers."""
    data = Path(path).read_bytes()
    info = sf.info(path)
    return data, {
        "method": "original file bytes",
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "num_frames": info.frames,
    }


def load_mono_float(path: str | Path, target_rate: int | None = None) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Decode to float32 mono, resampling with soxr (HQ) if `target_rate` differs from the file's rate."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    details: dict[str, Any] = {"source_sample_rate": rate, "source_channels": audio.shape[1]}
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
        details["downmix"] = "mean of channels"
    mono = audio[:, 0]
    if target_rate is not None and target_rate != rate:
        mono = resample(mono, rate, target_rate)
        details["resample"] = {"library": "soxr", "quality": "HQ", "from": rate, "to": target_rate}
        rate = target_rate
    details.update({"sample_rate": rate, "num_frames": int(mono.shape[0]), "duration_seconds": mono.shape[0] / rate})
    return mono, rate, details


def resample(audio: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    import soxr

    return soxr.resample(audio, rate_in, rate_out, quality="HQ").astype(np.float32, copy=False)
