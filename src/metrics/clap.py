"""MS-CLAP 2023 embeddings for CLAP similarity and MACE (E10).

MS-CLAP 2023 takes 7 s of 44.1 kHz audio; msclap itself scores one random 7 s crop of a longer clip. As in the MACE
paper, the whole clip is embedded instead: it is cut into consecutive 7 s chunks, a short final chunk is filled by
repeating it (as msclap does for short clips), and the chunk embeddings are averaged, weighted by their real duration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def chunk_audio(samples: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive chunks of `chunk_size` samples (k, chunk_size) and each chunk's real length in samples (k,)."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError(f"Expected non-empty mono audio, got shape {samples.shape}")
    chunks, lengths = [], []
    for start in range(0, samples.size, chunk_size):
        piece = samples[start : start + chunk_size]
        lengths.append(piece.size)
        if piece.size < chunk_size:
            piece = np.tile(piece, -(-chunk_size // piece.size))[:chunk_size]
        chunks.append(piece)
    return np.stack(chunks), np.asarray(lengths, dtype=np.float64)


def weighted_mean(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    return (np.asarray(embeddings, dtype=np.float64) * weights[:, None]).sum(axis=0) / weights.sum()


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of matching rows of `a` and `b`."""
    return (normalize(a) * normalize(b)).sum(axis=-1)


def mean_reference_cosine(candidates: np.ndarray, references: Sequence[np.ndarray]) -> np.ndarray:
    """Per item, the mean cosine between the candidate embedding and each of its reference embeddings."""
    return np.array(
        [(normalize(refs) @ normalize(cand)).mean() for cand, refs in zip(candidates, references, strict=True)]
    )


def read_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Read a WAV as float32 mono (channel mean); the file must already be at `sample_rate` (no resampling)."""
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"{path}: {rate} Hz, expected {sample_rate} Hz")
    return samples.mean(axis=1)


class ClapEmbedder:
    """MS-CLAP 2023 text and whole-clip audio embeddings (projected, not normalized), cached per text and per file."""

    def __init__(self, weights: str | Path, settings: dict, batch_size: int = 32) -> None:
        from msclap import CLAP

        self.model = CLAP(model_fp=str(weights), version=settings["version"], use_cuda=False)
        self.sample_rate = settings["sample_rate"]
        self.chunk_size = int(settings["chunk_seconds"] * self.sample_rate)
        self.batch_size = batch_size
        self._text: dict[str, np.ndarray] = {}
        self._audio: dict[str, np.ndarray] = {}

    def text(self, texts: Sequence[str]) -> np.ndarray:
        if new := sorted(set(texts) - set(self._text)):
            for start in range(0, len(new), self.batch_size):
                batch = new[start : start + self.batch_size]
                embeddings = self.model.get_text_embeddings(batch).detach().cpu().numpy()
                self._text.update(zip(batch, embeddings.astype(np.float64), strict=True))
        return np.stack([self._text[text] for text in texts])

    def audio(self, paths: Sequence[str | Path]) -> np.ndarray:
        import torch

        for path in map(str, paths):
            if path in self._audio:
                continue
            chunks, lengths = chunk_audio(read_mono(path, self.sample_rate), self.chunk_size)
            with torch.no_grad():
                embeddings = self.model.clap.audio_encoder(torch.from_numpy(chunks))[0].cpu().numpy()
            self._audio[path] = weighted_mean(embeddings, lengths)
        return np.stack([self._audio[str(path)] for path in paths])

    def preload_audio(self, embeddings: dict[str, np.ndarray]) -> None:
        """Reuse audio embeddings computed earlier (e.g. from a cache file), keyed by path."""
        self._audio.update(embeddings)

    def audio_embeddings(self) -> dict[str, np.ndarray]:
        """Every audio embedding computed or preloaded so far, keyed by path."""
        return dict(self._audio)


def load_audio_cache(path: str | Path, key: str) -> dict[str, np.ndarray]:
    """Audio embeddings keyed by audio SHA-256, or {} if the file is missing or was written with a different `key`."""
    path = Path(path)
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        if str(data["key"]) != key:
            return {}
        return dict(zip(data["sha256"].tolist(), data["embeddings"], strict=True))


def save_audio_cache(path: str | Path, key: str, embeddings: dict[str, np.ndarray]) -> Path:
    """Write the cache in one step (temporary file, then rename). `key` names the settings the embeddings depend on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shas = sorted(embeddings)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, key=np.array(key), sha256=np.array(shas), embeddings=np.stack([embeddings[s] for s in shas]))
    tmp.replace(path)
    return path
