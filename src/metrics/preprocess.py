"""Text preprocessing shared by every text metric (E10): PTB tokenization, as in aac-metrics and DCASE.

The PTB tokenizer lowercases, splits off and removes punctuation, and returns tokens joined by spaces. It works line by
line, so each distinct string is tokenized once and the result mapped back.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def check_texts(texts: Sequence[str], what: str) -> list[str]:
    """Problems that would break line-based tools or make a score meaningless: non-strings, empty or multi-line text."""
    errors = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            errors.append(f"{what}[{index}]: not a string ({type(text).__name__})")
        elif not text.strip():
            errors.append(f"{what}[{index}]: empty")
        elif "\n" in text or "\r" in text:
            errors.append(f"{what}[{index}]: contains a line break")
    return errors


def subword_lengths(texts: Sequence[str], tokenizer_name: str, suffix: str = "") -> list[int]:
    """Subword count of each text (special tokens included) under a Hugging Face tokenizer, as a model would see it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [len(tokenizer(text + suffix, add_special_tokens=True)["input_ids"]) for text in texts]


def truncation_summary(lengths: Sequence[int], limit: int) -> dict[str, int]:
    """How many inputs exceed a model's maximum length (they would be truncated or rejected)."""
    return {"limit": limit, "max_length": max(lengths), "num_over_limit": sum(n > limit for n in lengths)}


def ptb_tokenize(texts: Sequence[str], *, java: str | Path, cache_path: str | Path, tmp_path: str | Path) -> list[str]:
    """PTB-tokenize `texts` with one Java call; raise if any input or output is invalid."""
    from aac_metrics.utils.tokenization import preprocess_mono_sents

    if errors := check_texts(texts, "text"):
        raise ValueError(f"Cannot tokenize: {errors[:5]}")
    unique = sorted(set(texts))
    tokenized = preprocess_mono_sents(unique, cache_path=str(cache_path), java_path=str(java), tmp_path=str(tmp_path))
    if len(tokenized) != len(unique):
        raise RuntimeError(f"PTB tokenizer returned {len(tokenized)} lines for {len(unique)} inputs")
    mapping = dict(zip(unique, tokenized, strict=True))
    if empty := [text for text, result in mapping.items() if not result.strip()]:
        raise ValueError(f"{len(empty)} texts are empty after PTB tokenization, e.g. {empty[0]!r}")
    return [mapping[text] for text in texts]
