"""BERTScore F1 per (candidate, reference) pair, with torchmetrics' reordering bug corrected.

aac-metrics 0.5.5 computes BERTScore through torchmetrics' `bert_score` (1.9.0, the latest release). That function
embeds the predictions and the targets in length-sorted order and then indexes the embeddings with the sorting
permutation instead of its inverse (torchmetrics/functional/text/bert.py, `embeddings[sorting_indices]`). Predictions
and targets are sorted separately, so in any call with more than one pair most candidates are scored against another
pair's reference. Scores are only right for a single pair, or when both permutations are their own inverses.

This module keeps torchmetrics' own tokenization (`TextDataset`), embedding (`_get_embeddings_and_idf_scale`: the
model's last hidden layer, L2-normalized, special tokens and padding masked) and greedy matching
(`_get_precision_recall_f1`), and restores the original order with the inverse permutation. Each distinct sentence is
embedded once. Without IDF weighting the result does not depend on which other sentences share the call, so it
matches torchmetrics run on one pair at a time (up to floating-point noise from batch padding).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _embed(texts: list[str], model: Any, tokenizer: Any, device: Any, batch_size: int, max_length: int):
    """Embeddings and token weights of `texts`, in the order given."""
    import torch
    from torch.utils.data import DataLoader
    from torchmetrics.functional.text.bert import _get_embeddings_and_idf_scale
    from torchmetrics.functional.text.helper_embedding_metric import TextDataset

    dataset = TextDataset(texts, tokenizer, max_length, idf=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    embeddings, scale = _get_embeddings_and_idf_scale(loader, dataset.max_length, model, device, None, False, False)
    inverse = torch.argsort(dataset.sorting_indices)  # sorted position -> original position
    return embeddings[inverse], scale[inverse]


def bertscore_pairs(
    candidates: list[str],
    references: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    max_length: int,
    chunk: int = 512,
) -> np.ndarray:
    """F1 of every (candidates[i], references[i]) pair (no IDF, no rescaling)."""
    import torch
    from torchmetrics.functional.text.bert import _get_precision_recall_f1

    if len(candidates) != len(references):
        raise ValueError(f"{len(candidates)} candidates vs {len(references)} references")
    if not candidates:
        return np.zeros(0)
    model.eval()
    model.to(device)
    unique_candidates, candidate_index = np.unique(np.array(candidates, dtype=object), return_inverse=True)
    unique_references, reference_index = np.unique(np.array(references, dtype=object), return_inverse=True)
    cand_emb, cand_scale = _embed(list(unique_candidates), model, tokenizer, device, batch_size, max_length)
    ref_emb, ref_scale = _embed(list(unique_references), model, tokenizer, device, batch_size, max_length)
    scores = []
    for start in range(0, len(candidates), chunk):
        c = torch.as_tensor(candidate_index[start : start + chunk])
        r = torch.as_tensor(reference_index[start : start + chunk])
        _, _, f1 = _get_precision_recall_f1(cand_emb[c], ref_emb[r], cand_scale[c], ref_scale[r])
        scores.append(f1.reshape(-1).double().numpy())
    return np.concatenate(scores)


def bertscore_max(
    candidates: list[str],
    mult_references: list[list[str]],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Per candidate, the maximum F1 over its references (aac-metrics' `reduction="max"`)."""
    pairs = [
        (candidate, reference)
        for candidate, refs in zip(candidates, mult_references, strict=True)
        for reference in refs
    ]
    if any(not refs for refs in mult_references):
        raise ValueError("Every candidate needs at least one reference")
    f1 = bertscore_pairs([p[0] for p in pairs], [p[1] for p in pairs], model, tokenizer, device, batch_size, max_length)
    bounds = np.cumsum([0, *(len(refs) for refs in mult_references)])
    return np.array([f1[start:stop].max() for start, stop in zip(bounds[:-1], bounds[1:], strict=True)])
