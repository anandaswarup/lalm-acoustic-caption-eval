"""Averaging over held-out positions, the leave-one-out checks and the calibration report (E13). No Java or models."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    calibration_checks,
    closest_reference_overlap,
    duplicate_held_out,
    overlap_keep_masks,
    overlap_rows,
    overlap_sample,
    overlap_summary,
    report_markdown,
)
from src.evaluation.scoring import ScoreSet
from src.evaluation.summary import ALL_RECORDINGS, POSITION_MEAN, average_subsets, masked_average, summary_rows
from src.metrics.registry import Score

IDS = ["s0", "s1", "s2"]
REFS = {sid: [f"{sid} caption {k}" for k in range(1, 6)] for sid in IDS}


def position_sets() -> list[ScoreSet]:
    sets = []
    for k in range(1, 6):
        items = np.array([0.1, 0.2, 0.3]) * k
        scores = {"bleu_1": Score(0.5 * k, items), "cider_d": Score(float(items.mean()), items)}
        sets.append(ScoreSet("human", "human_loo", f"drop_{k}", list(IDS), scores))
    return sets


def test_average_subsets():
    averaged = average_subsets(position_sets())
    assert (averaged.system, averaged.condition, averaged.subset, averaged.sample_ids) == (
        "human", "human_loo", POSITION_MEAN, IDS
    )  # fmt: skip
    np.testing.assert_allclose(averaged.scores["bleu_1"].items, [0.3, 0.6, 0.9])  # mean of k = 3
    assert averaged.scores["bleu_1"].corpus == pytest.approx(1.5)  # mean of the corpus values, not of the items
    assert averaged.scores["cider_d"].corpus == pytest.approx(averaged.scores["cider_d"].items.mean())
    sets = position_sets()
    with pytest.raises(ValueError, match="same system"):
        average_subsets([sets[0], ScoreSet("gpt", "human_loo", "drop_2", IDS, sets[1].scores)])
    with pytest.raises(ValueError, match="Duplicate"):
        average_subsets([sets[0], sets[0]])
    with pytest.raises(ValueError, match="different metrics"):
        average_subsets([sets[0], ScoreSet("human", "human_loo", "drop_2", IDS, {"bleu_1": sets[1].scores["bleu_1"]})])


def test_calibration_checks_pass_on_valid_references():
    checks = calibration_checks(REFS, 5)
    assert len(checks) == 2 + 5 + 1 and all(c["passed"] for c in checks)


def test_calibration_checks_fail():
    short = {**REFS, "s1": REFS["s1"][:4]}
    checks = calibration_checks(short, 5)
    assert not checks[0]["passed"] and "s1: 4" in checks[0]["detail"]
    assert len(checks) == 2  # the split checks need five references everywhere
    empty = {**REFS, "s2": REFS["s2"][:4] + [" "]}
    assert not calibration_checks(empty, 5)[1]["passed"]


def test_duplicate_held_out():
    tokenized = {"s0": ["a dog barks", "rain", "a dog barks", "wind", "cars"], "s1": ["x", "y", "z", "w", "v"]}
    assert duplicate_held_out(tokenized) == ["s0:1", "s0:3"]


TOKENIZED = {
    "s0": ["a dog barks loudly", "a dog barks", "rain falls", "wind blows hard", "cars pass by"],
    "s1": ["x y", "y z", "z w", "w v", "v u"],
    "s2": ["a b c d", "a b c d", "e f", "g h", "i j"],
}


def test_closest_reference_overlap_and_masks():
    overlaps = closest_reference_overlap(TOKENIZED)
    assert overlaps["s0"][:3] == [pytest.approx(0.75), pytest.approx(0.75), 0.0]  # {a,dog,barks} / {a,dog,barks,loudly}
    assert overlaps["s1"] == [pytest.approx(1 / 3)] * 5
    assert overlaps["s2"][:2] == [1.0, 1.0]
    masks = overlap_keep_masks(overlaps, IDS, [f"drop_{k}" for k in range(1, 6)], 0.6)
    assert masks["drop_1"].tolist() == [False, True, False]  # position 1 of s0 and s2 has a near-copy
    assert masks["drop_3"].tolist() == [True, True, True]
    summary = overlap_summary(overlaps, [0.6, 0.8])
    assert summary["held_out_captions"] == 15
    assert summary["thresholds"][0] == {"threshold": 0.6, "at_or_above": 4, "share": round(4 / 15, 4), "recordings": 2}
    assert summary["thresholds"][1]["at_or_above"] == 2
    assert overlap_sample(0.6) == "closest_overlap_below_0.6"
    assert overlap_rows({"s0": [0.5]}) == [
        {"sample_id": "s0", "position": 1, "subset": "drop_1", "closest_reference_jaccard": "0.5"}
    ]


def test_masked_average():
    sets = position_sets()
    keep = {f"drop_{k}": np.array([k != 1, True, False]) for k in range(1, 6)}  # s2 loses every position
    kept = masked_average(sets, keep)
    assert kept.sample_ids == ["s0", "s1"] and kept.subset == POSITION_MEAN
    np.testing.assert_allclose(kept.scores["cider_d"].items, [0.1 * 3.5, 0.2 * 3.0])  # s0: mean of k = 2..5
    assert np.isnan(kept.scores["bleu_1"].corpus)
    assert kept.scores["cider_d"].corpus == pytest.approx(kept.scores["cider_d"].items.mean())
    everything = masked_average(sets, {f"drop_{k}": np.ones(3, dtype=bool) for k in range(1, 6)})
    np.testing.assert_allclose(everything.scores["bleu_1"].items, average_subsets(sets).scores["bleu_1"].items)
    with pytest.raises(ValueError, match="Mask shape"):
        masked_average(sets, {f"drop_{k}": np.ones(2, dtype=bool) for k in range(1, 6)})


def test_report_markdown():
    sets = position_sets()
    averaged = average_subsets(sets)
    summary = [row for s in [*sets, averaged] for row in summary_rows([s], ALL_RECORDINGS, 100, 0.95, 42)]
    overlaps = closest_reference_overlap(TOKENIZED)
    for threshold in (0.6, 0.8):
        kept = masked_average(sets, overlap_keep_masks(overlaps, IDS, [f"drop_{k}" for k in range(1, 6)], threshold))
        summary += summary_rows([kept], overlap_sample(threshold), 100, 0.95, 42)
    report = {
        "run_info": {"git_commit": "abc", "git_dirty": False, "timestamp_utc": "now"},
        "subsets": [f"drop_{k}" for k in range(1, 6)],
        "smoke_test_limit": 3,
        "metrics": ["bleu_1", "cider_d"],
        "inputs": {"recordings": 3, "held_out_captions": 15, "references_per_subset": 4,
                   "references": {"path": "r.json", "sha256": "1"}},
        "checks": calibration_checks(REFS, 5),
        "duplicates": [],
        "overlap": overlap_summary(overlaps, [0.6, 0.8]),
        "outputs": {"overlap": "human_loo_overlap.csv"},
        "bootstrap": {"num_resamples": 100, "confidence": 0.95, "seed": 42},
        "summary": summary,
        "timing": {"seconds": {"bleu": 1.0}, "calls": {"bleu": 5}, "load_models_seconds": 3.0,
                   "scoring_seconds": 60.0, "clap_cache_hits": 3},
        "limitations": ["note"],
    }  # fmt: skip
    text = report_markdown(report)
    assert "Smoke test on the first 3 recordings" in text
    assert "| BLEU-1 | lexical | 1.5000 (0.6000) [" in text
    assert "| 0.5000 | 1.0000 | 1.5000 | 2.0000 | 2.5000 | 2.0000 |" in text  # per-position corpus values and range
    assert "| >= 0.6 | 4 | 26.7% | 2 |" in text
    assert "Closest overlap < 0.6 (3)" in text and "Closest overlap < 0.8 (3)" in text
    assert "nan" not in text
