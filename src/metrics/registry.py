"""Metric names and metadata (E10). The order here is the reporting order."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Score(NamedTuple):
    """A metric's corpus value and its per-item scores, in input order."""

    corpus: float
    items: np.ndarray


class Timer:
    """Wall-clock seconds and call counts per named step, accumulated across calls (for the implementation report)."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1


def check_scores(scores: dict[str, Score], num_items: int) -> list[str]:
    """Problems with metric outputs: wrong number of items or non-finite values."""
    errors = []
    for name, score in scores.items():
        if score.items.shape != (num_items,):
            errors.append(f"{name}: {score.items.shape} item scores for {num_items} items")
        elif not np.isfinite(score.items).all():
            errors.append(f"{name}: {int((~np.isfinite(score.items)).sum())} non-finite item scores")
        if not np.isfinite(score.corpus):
            errors.append(f"{name}: non-finite corpus score {score.corpus}")
    return errors


@dataclass(frozen=True)
class Metric:
    name: str
    family: str  # lexical, semantic, audio_aware, or auxiliary (a component reported for transparency)
    label: str
    # False when the corpus value is not the mean of the per-item scores (corpus BLEU and METEOR statistics).
    corpus_is_item_mean: bool = True
    needs_audio: bool = False


METRICS: tuple[Metric, ...] = (
    Metric("bleu_1", "lexical", "BLEU-1", corpus_is_item_mean=False),
    Metric("bleu_2", "lexical", "BLEU-2", corpus_is_item_mean=False),
    Metric("bleu_3", "lexical", "BLEU-3", corpus_is_item_mean=False),
    Metric("bleu_4", "lexical", "BLEU-4", corpus_is_item_mean=False),
    Metric("meteor", "lexical", "METEOR", corpus_is_item_mean=False),
    Metric("cider_d", "lexical", "CIDEr-D"),
    Metric("spice", "semantic", "SPICE"),
    Metric("spider", "semantic", "SPIDEr"),
    Metric("spider_fl", "semantic", "SPIDEr-FL"),
    Metric("fense", "semantic", "FENSE"),
    Metric("bertscore_f1", "semantic", "BERTScore-F1"),
    Metric("bertscore_f1_rescaled", "semantic", "BERTScore-F1 (rescaled)"),
    Metric("clap_similarity", "audio_aware", "CLAP similarity", needs_audio=True),
    Metric("mace", "audio_aware", "MACE", needs_audio=True),
    Metric("mace_text", "auxiliary", "MACE (text only)"),
    Metric("mace_audio", "auxiliary", "MACE (audio only)", needs_audio=True),
    Metric("clap_text_similarity", "auxiliary", "CLAP text-text similarity"),
    Metric("sbert_similarity", "auxiliary", "SBERT similarity (FENSE without penalty)"),
    Metric("fluency_error", "auxiliary", "Fluency error rate (threshold 0.9)"),
    Metric("fluency_error_mace", "auxiliary", "Fluency error rate (threshold 0.97)"),
)
BY_NAME = {metric.name: metric for metric in METRICS}
REFERENCE_FREE = {"clap_similarity", "mace_audio", "fluency_error", "fluency_error_mace"}
