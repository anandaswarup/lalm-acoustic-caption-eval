"""Blinded qualitative review sample (E17). No Java or models."""

import numpy as np
import pytest

from src.analysis.qualitative import (
    ANNOTATION_COLUMNS,
    REVIEW_COLUMNS,
    annotation_template,
    blind,
    blinding_problems,
    review_rows,
    sample_comparison,
    select_sample,
)

IDS = [f"clotho_eval_{i:04d}" for i in range(200)]
MODELS = ["gpt_audio_1_5", "gemini_3_8_flash", "qwen2_5_omni", "audio_flamingo_3"]


def inputs() -> tuple[dict, dict]:
    references = {sid: [f"ref {k} of {sid}" for k in range(5)] for sid in IDS}
    captions = {m: {sid: {"caption": f"caption {sid}", "normalization": "none"} for sid in IDS} for m in MODELS}
    return references, captions


def test_sample_is_seeded_and_order_independent():
    sample = select_sample(IDS, 50, 42)
    assert len(sample) == len(set(sample)) == 50 and sample == sorted(sample)
    assert select_sample(list(reversed(IDS)), 50, 42) == sample
    assert select_sample(IDS, 50, 43) != sample


def test_blinding_labels_are_a_permutation_per_recording():
    sample = select_sample(IDS, 50, 42)
    key = blind(sample, MODELS, 42)
    assert len(key) == 200 and key == blind(sample, MODELS, 42)
    by_code: dict[str, list[dict]] = {}
    for row in key:
        by_code.setdefault(row["recording_code"], []).append(row)
    assert len(by_code) == 50 and sorted({r["sample_id"] for r in key}) == sample
    for rows in by_code.values():
        assert [r["caption_label"] for r in rows] == list("ABCD")
        assert sorted(r["model_key"] for r in rows) == sorted(MODELS)
        assert len({r["sample_id"] for r in rows}) == 1
    # model orders and recording order are randomized, not fixed
    assert len({tuple(r["model_key"] for r in rows) for rows in by_code.values()}) > 1
    assert [rows[0]["sample_id"] for rows in by_code.values()] != sample
    assert len({r["item_id"] for r in key}) == 200


def test_review_rows_hold_no_identifying_fields():
    references, captions = inputs()
    key = blind(select_sample(IDS, 50, 42), MODELS, 42)
    review, full_key = review_rows(key, references, captions)
    assert all(tuple(row) == REVIEW_COLUMNS for row in review)
    assert all(row["normalization"] == "none" for row in full_key)
    assert blinding_problems(review, MODELS, ["dog.wav"]) == []
    references[key[0]["sample_id"]][0] = "recorded with gpt_audio_1_5 near dog.wav"
    review, _ = review_rows(key, references, captions)
    assert len(blinding_problems(review, MODELS, ["dog.wav"])) == 8  # 4 items x 2 terms
    template = annotation_template(review)
    assert all(tuple(row) == ANNOTATION_COLUMNS and row["category"] == "" for row in template)


def test_sample_comparison():
    values = {"g": {"x": np.arange(200.0)}}
    (row,) = sample_comparison(IDS[:10], IDS, values)
    assert row["sample_mean"] == pytest.approx(4.5) and row["population_mean"] == pytest.approx(99.5)
    assert row["standardized_difference"] < 0
