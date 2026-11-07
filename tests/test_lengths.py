"""Caption-length statistics, model-human comparison and length-score associations (E14). No Java or models."""

import math

import numpy as np
import pytest

from src.analysis.lengths import (
    HUMAN,
    PER_CAPTION_COLUMNS,
    SUMMARY_COLUMNS,
    association_rows,
    human_lengths,
    length_stats,
    model_lengths,
    per_caption_rows,
    report_markdown,
    spearman_rows,
    summary_rows,
    word_count,
)

IDS = [f"s{i}" for i in range(30)]
BOOT = (200, 0.95, 42)


def references(seed: int = 0) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    return {sid: [" ".join(["w"] * int(n)) for n in rng.integers(8, 15, 5)] for sid in IDS}


def captions(words: list[int]) -> dict[str, str]:
    return {sid: " ".join(["x"] * n) for sid, n in zip(IDS, words, strict=True)}


def test_word_count_splits_on_any_whitespace():
    assert word_count("  a dog\tbarks\nloudly. ") == 4
    assert word_count('{"a": "b c"}') == 3


def test_human_lengths_compare_each_caption_with_the_other_references():
    refs = {"s0": ["a b", "a b c d", "a", "a b c", "a b c d e f"]}
    item = human_lengths(refs, ["s0"])
    assert item.words.tolist() == [[2, 4, 1, 3, 6]]
    assert item.reference_mean[0, 0] == pytest.approx((4 + 1 + 3 + 6) / 4)
    assert item.reference_max[0].tolist() == [6, 6, 6, 6, 4]
    rows = per_caption_rows([item])
    assert [r["position"] for r in rows] == [1, 2, 3, 4, 5]
    assert set(rows[0]) == set(PER_CAPTION_COLUMNS)


def test_model_lengths_use_all_references():
    refs = references()
    item = model_lengths("m", captions([10] * len(IDS)), refs, IDS)
    assert item.words.shape == (len(IDS), 1)
    assert item.reference_mean[3, 0] == pytest.approx(np.mean([word_count(r) for r in refs["s3"]]))
    assert item.restrict(["s3", "s1"]).words.shape == (2, 1)
    with pytest.raises(ValueError):
        item.restrict(["missing"])


def test_length_stats():
    stats = length_stats(np.array([1, 2, 3, 4, 10]))
    assert stats["mean"] == 4 and stats["median"] == 3 and stats["iqr"] == 2
    assert stats["sd"] == pytest.approx(np.std([1, 2, 3, 4, 10], ddof=1))


def test_summary_ratio_and_difference():
    refs = references()
    human = human_lengths(refs, IDS)
    double = np.rint(2 * human.words.mean(axis=1)).astype(int)
    model = model_lengths("m", captions(double.tolist()), refs, IDS)
    rows = {r["system"]: r for r in summary_rows([model, human], "all", *BOOT)}
    assert all(list(r) == list(SUMMARY_COLUMNS) for r in rows.values())
    ratio = float(rows["m"]["ratio_to_human"])
    assert ratio == pytest.approx(double.mean() / human.words.mean())
    assert float(rows["m"]["ratio_ci_low"]) <= ratio <= float(rows["m"]["ratio_ci_high"])
    assert float(rows["m"]["difference_mean"]) == pytest.approx((double - human.words.mean(axis=1)).mean())
    assert float(rows["m"]["share_longer_than_reference_mean"]) == 1.0
    assert rows[HUMAN]["ratio_to_human"] == "" and rows[HUMAN]["num_captions"] == 5 * len(IDS)


def test_spearman_matches_rank_correlation_with_ties():
    x = np.array([1, 2, 2, 3, 5, 4])
    y = np.array([2, 1, 4, 3, 6, 5])
    rx, ry = (np.argsort(np.argsort(v)).astype(float) for v in (x, y))
    rx[[1, 2]] = 1.5  # the tied pair
    assert spearman_rows(x, y)[0] == pytest.approx(np.corrcoef(rx, ry)[0, 1])
    assert spearman_rows(-x, y)[0] == pytest.approx(-spearman_rows(x, y)[0])
    assert math.isnan(spearman_rows(np.ones(4), np.arange(4))[0])


def test_associations_detect_a_length_effect_and_keep_recordings_together():
    refs = references()
    human = human_lengths(refs, IDS)
    rng = np.random.default_rng(1)
    scores = {"cider_d": human.words + rng.normal(0, 0.5, human.words.shape), "spice": rng.normal(size=(30, 5))}
    rows = {(r["predictor"], r["metric"]): r for r in association_rows(human, scores, "all", *BOOT)}
    assert len(rows) == 4
    strong = rows[("words", "cider_d")]
    assert float(strong["spearman"]) > 0.8 and strong["excludes_zero"]
    assert strong["num_pairs"] == 150 and strong["num_recordings"] == 30
    with pytest.raises(ValueError):
        association_rows(human, {"cider_d": np.zeros((30, 1))}, "all", *BOOT)


def test_report_has_no_caption_text():
    refs = references()
    model = model_lengths("m", {sid: "secretword " * 12 for sid in IDS}, refs, IDS)
    lengths = [model, human_lengths(refs, IDS)]
    scores = {"cider_d": np.arange(30.0)[:, None]}
    human_scores = {"cider_d": np.arange(150.0).reshape(30, 5)}
    summary = summary_rows(lengths, "all", *BOOT) + summary_rows([x.restrict(IDS[:20]) for x in lengths], "v", *BOOT)
    associations = association_rows(model, scores, "all", *BOOT)
    associations += association_rows(lengths[1], human_scores, "all", *BOOT)
    info = {"path": "p", "sha256": "0"}
    report = {
        "run_info": {"git_commit": "abc", "git_dirty": False, "timestamp_utc": "t"},
        "display_names": {"m": "Model"},
        "word_definition": "whitespace",
        "samples": {"all": "all", "valid": "v"},
        "association_metrics": ["cider_d"],
        "inputs": {
            "recordings": 30,
            "references_per_recording": 5,
            "captions": info,
            "references": info,
            "model_scores": info,
            "human_scores": info,
        },  # fmt: skip
        "bootstrap": {"num_resamples": 200, "confidence": 0.95, "seed": 42},
        "summary": summary,
        "associations": associations,
        "outputs": {"summary": "x.csv"},
        "limitations": ["none"],
    }
    text = report_markdown(report)
    assert "secretword" not in text and "w w" not in text
    assert "| Model | 30 | 12.00 |" in text and "CIDEr-D" in text
