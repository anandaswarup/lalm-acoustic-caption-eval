"""Evaluation conditions and reference subsets (E12-E14).

- five_reference: each model caption against all five references (subset `all`).
- four_reference: each model caption against subset `drop_k`, the four references left after removing CSV position k.
- human_loo: the human caption at position k against subset `drop_k`, the other four.

`drop_k` is the same reference set in E13 and E14, so model and human scores are compared under identical references.
Positions are 1-based and follow the original Clotho CSV order kept in the references file.
"""

from __future__ import annotations

FIVE_REFERENCE, FOUR_REFERENCE, HUMAN_LOO = "five_reference", "four_reference", "human_loo"
CONDITIONS = (FIVE_REFERENCE, FOUR_REFERENCE, HUMAN_LOO)
ALL = "all"


def subsets(condition: str, num_references: int = 5) -> list[str]:
    if condition == FIVE_REFERENCE:
        return [ALL]
    if condition in (FOUR_REFERENCE, HUMAN_LOO):
        return [f"drop_{k}" for k in range(1, num_references + 1)]
    raise ValueError(f"Unknown condition {condition!r} (expected one of {CONDITIONS})")


def dropped_index(subset: str) -> int | None:
    """0-based index of the reference removed by `subset`, or None for `all`."""
    if subset == ALL:
        return None
    if subset.startswith("drop_") and subset[5:].isdigit() and int(subset[5:]) >= 1:
        return int(subset[5:]) - 1
    raise ValueError(f"Unknown reference subset {subset!r}")


def subset_references(references: dict[str, list[str]], subset: str) -> dict[str, list[str]]:
    index = dropped_index(subset)
    if index is None:
        return {sample_id: list(refs) for sample_id, refs in references.items()}
    return {sample_id: refs[:index] + refs[index + 1 :] for sample_id, refs in references.items()}


def held_out_captions(references: dict[str, list[str]], subset: str) -> dict[str, str]:
    """The human caption removed by `subset` for each recording (the human_loo candidate)."""
    index = dropped_index(subset)
    if index is None:
        raise ValueError("The `all` subset holds out no caption")
    return {sample_id: refs[index] for sample_id, refs in references.items()}


def check_split(references: dict[str, list[str]], subset: str) -> list[str]:
    """Verify by position that each subset has one reference fewer and that it is exactly the held-out one.

    Checked by position, not by text: two annotators can write the same caption, and the duplicate is a valid reference.
    """
    index = dropped_index(subset)
    if index is None:
        raise ValueError("The `all` subset holds out no caption")
    errors = []
    for sample_id, refs in references.items():
        if index >= len(refs):
            errors.append(f"{sample_id}: {subset} drops position {index + 1} of {len(refs)} references")
            continue
        single = {sample_id: refs}
        kept, held_out = subset_references(single, subset)[sample_id], held_out_captions(single, subset)[sample_id]
        if len(kept) != len(refs) - 1 or kept[:index] + [held_out] + kept[index:] != refs:
            errors.append(f"{sample_id}: {subset} does not split the references at position {index + 1}")
    return errors
