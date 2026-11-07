"""CLAP similarity and MACE (E10).

MACE follows the paper (Dixit et al., arXiv 2411.00321): 0.5 * (audio-text + mean text-text CLAP cosine), multiplied by
(1 - penalty * error) where error flags a fluency error (probability > threshold). aac-metrics 0.5.5 differs: its
default is text-only, it embeds one random 7 s crop, and its per-item MACE holds the corpus mean. MACE-text and
MACE-audio are the paper's ablations, reported as auxiliary metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.metrics.clap import cosine_rows, mean_reference_cosine
from src.metrics.reference_based import flag_errors, penalize
from src.metrics.registry import Score, check_scores


def mace_scores(
    audio_text: np.ndarray, text_text: np.ndarray, error_probs: np.ndarray, *, error_threshold: float, penalty: float
) -> dict[str, Score]:
    errors = flag_errors(error_probs, error_threshold)
    combined = 0.5 * (np.asarray(audio_text) + np.asarray(text_text))
    items = {
        "clap_similarity": np.asarray(audio_text, dtype=np.float64),
        "clap_text_similarity": np.asarray(text_text, dtype=np.float64),
        "mace": penalize(combined, errors, penalty),
        "mace_text": penalize(text_text, errors, penalty),
        "mace_audio": penalize(audio_text, errors, penalty),
        "fluency_error_mace": errors,
    }
    return {name: Score(float(values.mean()), values) for name, values in items.items()}


def audio_aware_scores(
    candidate_embeddings: np.ndarray,
    reference_embeddings: Sequence[np.ndarray],
    audio_embeddings: np.ndarray,
    error_probs: np.ndarray,
    settings: dict,
) -> dict[str, Score]:
    """CLAP similarity, MACE and its variants from precomputed CLAP embeddings and fluency-error probabilities."""
    scores = mace_scores(
        cosine_rows(audio_embeddings, candidate_embeddings),
        mean_reference_cosine(candidate_embeddings, reference_embeddings),
        error_probs,
        error_threshold=settings["error_threshold"],
        penalty=settings["penalty"],
    )
    if errors := check_scores(scores, len(candidate_embeddings)):
        raise RuntimeError(f"Invalid audio-aware scores: {errors}")
    return scores
