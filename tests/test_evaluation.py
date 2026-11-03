"""Evaluation conditions, input checks, score tables and the bootstrap (E11). No Java or models needed."""

import csv
import json

import numpy as np
import pytest

from src.evaluation import references as conditions
from src.evaluation.aggregate import bootstrap_difference, bootstrap_indices, bootstrap_mean, percentile_interval
from src.evaluation.inputs import audio_paths, check_condition_inputs, load_model_captions
from src.evaluation.scoring import CORPUS_COLUMNS, SCORE_COLUMNS, ScoreSet, corpus_rows, score_rows, write_csv
from src.metrics.registry import Score

REFS = {
    "s0": ["r0a", "r0b", "r0c", "r0d", "r0e"],
    "s1": ["r1a", "same", "r1c", "same", "r1e"],  # two annotators wrote the same caption
}


def test_subsets():
    assert conditions.subsets("five_reference") == ["all"]
    expected = ["drop_1", "drop_2", "drop_3", "drop_4", "drop_5"]
    assert conditions.subsets("four_reference") == conditions.subsets("human_loo") == expected
    with pytest.raises(ValueError):
        conditions.subsets("three_reference")
    for bad in ("drop_0", "drop_x", "keep_1"):
        with pytest.raises(ValueError):
            conditions.dropped_index(bad)


def test_four_reference_subsets_and_held_out_captions():
    for k in range(1, 6):
        subset = f"drop_{k}"
        kept, held_out = conditions.subset_references(REFS, subset), conditions.held_out_captions(REFS, subset)
        for sid, refs in REFS.items():
            assert len(kept[sid]) == 4
            assert held_out[sid] == refs[k - 1]
            assert kept[sid] == refs[: k - 1] + refs[k:]
        assert conditions.check_split(REFS, subset) == []
    # Every position is held out exactly once across the five subsets.
    held = [conditions.held_out_captions(REFS, f"drop_{k}")["s0"] for k in range(1, 6)]
    assert held == REFS["s0"]
    # A duplicate caption stays a valid reference when its twin is held out.
    assert "same" in conditions.subset_references(REFS, "drop_2")["s1"]
    assert conditions.subset_references(REFS, "all") == REFS


def test_check_split_flags_short_reference_lists():
    assert "drops position 5 of 4" in conditions.check_split({"s": ["a", "b", "c", "d"]}, "drop_5")[0]
    with pytest.raises(ValueError):
        conditions.check_split(REFS, "all")


def test_check_condition_inputs():
    ids = ["s0", "s1"]
    assert check_condition_inputs(ids, {"s0": "a dog", "s1": "rain"}, REFS, 5) == []
    errors = check_condition_inputs(ids + ["s2"], {"s0": "a dog", "s3": "x"}, REFS, 5)
    assert any("without a candidate" in e for e in errors)
    assert any("unknown recordings" in e for e in errors)
    assert any("without references" in e for e in errors)
    errors = check_condition_inputs(ids, {"s0": "", "s1": "a\nb"}, {"s0": REFS["s0"], "s1": ["x"] * 4}, 5)
    assert errors == ["candidate[0]: empty", "candidate[1]: contains a line break", "s1: 4 references, expected 5"]
    assert check_condition_inputs(["s0", "s0"], {"s0": "a"}, REFS, 5)[0] == "duplicate sample_ids"


def test_load_model_captions(tmp_path):
    path = tmp_path / "captions.jsonl"
    records = [
        {"sample_id": "s0", "model_key": "m1", "caption": "a dog", "caption_raw": '{"a dog"}'},
        {"sample_id": "s0", "model_key": "m2", "caption": "rain"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert load_model_captions(path) == {"m1": {"s0": "a dog"}, "m2": {"s0": "rain"}}
    path.write_text("".join(json.dumps(r) + "\n" for r in records + records[:1]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_model_captions(path)


def test_audio_paths_verify_checksums(tmp_path):
    from src.data.clotho import sha256sum

    (tmp_path / "a.wav").write_bytes(b"audio")
    manifest = [{"sample_id": "s0", "audio_path": "a.wav", "audio_sha256": sha256sum(tmp_path / "a.wav")}]
    assert audio_paths(manifest, tmp_path) == {"s0": tmp_path / "a.wav"}
    with pytest.raises(ValueError, match="does not match"):
        audio_paths([{**manifest[0], "audio_sha256": "0" * 64}], tmp_path)
    with pytest.raises(ValueError, match="not found"):
        audio_paths([{**manifest[0], "audio_path": "b.wav"}], tmp_path)


def test_bootstrap_is_seeded_and_paired():
    first, second = bootstrap_indices(50, 200, seed=42), bootstrap_indices(50, 200, seed=42)
    assert first.shape == (200, 50) and (first == second).all()
    assert not (first == bootstrap_indices(50, 200, seed=43)).all()
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 50)
    b = a - 0.1  # paired: every recording differs by exactly 0.1
    result = bootstrap_difference(a, b, first, 0.95)
    assert result["mean"] == pytest.approx(0.1)
    assert result["ci_low"] == pytest.approx(0.1) and result["ci_high"] == pytest.approx(0.1)
    ci = bootstrap_mean(a, first, 0.95)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    with pytest.raises(ValueError):
        bootstrap_difference(a, b[:10], first, 0.95)
    with pytest.raises(ValueError):
        bootstrap_mean(a[:10], first, 0.95)


def test_constant_scores_have_a_zero_width_interval():
    assert bootstrap_mean(np.full(20, 0.3), bootstrap_indices(20, 100, 1), 0.95) == pytest.approx(
        {"mean": 0.3, "ci_low": 0.3, "ci_high": 0.3}
    )
    assert percentile_interval(np.arange(101.0), 0.9) == pytest.approx((5.0, 95.0))


def test_score_tables(tmp_path):
    score_set = ScoreSet(
        "gpt", "four_reference", "drop_2", ["s0", "s1"], {"bleu_1": Score(0.55, np.array([0.5, 0.25]))}
    )
    scores = write_csv(score_rows(score_set), tmp_path / "scores.csv", SCORE_COLUMNS)
    corpus = write_csv(corpus_rows(score_set), tmp_path / "corpus.csv", CORPUS_COLUMNS)
    with scores.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["sample_id"], float(r["score"])) for r in rows] == [("s0", 0.5), ("s1", 0.25)]
    with corpus.open(encoding="utf-8") as stream:
        (row,) = list(csv.DictReader(stream))
    assert (float(row["corpus"]), float(row["item_mean"]), row["num_items"]) == (0.55, 0.375, "2")


def test_compare_runs_and_extrapolation():
    from src.evaluation.validation import compare_runs, corpus_table, extrapolate

    def run(value):
        return [ScoreSet("m", "five_reference", "all", ["s0", "s1"], {"spice": Score(value, np.array([0.1, value]))})]

    same = compare_runs(run(0.5), run(0.5))
    assert same == {"identical": True, "values_compared": 3, "max_abs_difference": 0.0, "differing": []}
    different = compare_runs(run(0.5), run(0.5 + 1e-12))
    assert not different["identical"] and different["differing"] == ["m/five_reference/all/spice"]
    assert different["max_abs_difference"] == pytest.approx(1e-12)
    assert corpus_table(run(0.5) + run(0.7), "five_reference") == {"m": {"spice": pytest.approx(0.6)}}

    estimate = extrapolate({"spice": 10.0, "bleu": 2.0, "load_models": 30.0}, 20, 1045)
    assert estimate["estimated_seconds"] == {"spice": 522.5, "bleu": 104.5}
    assert estimate["estimated_total_hours"] == pytest.approx(0.17)


def test_two_point_estimate():
    from src.evaluation.validation import two_point_estimate

    # 5 s fixed + 0.1 s per item: 7 s at 20 items, 13 s at 80 items.
    estimate = two_point_estimate({"spice": 7.0, "load_models": 3.0}, 20, {"spice": 13.0}, 80, 1045, 29)
    assert estimate["steps"] == {
        "spice": {"fixed_seconds": 5.0, "seconds_per_item": 0.1, "estimated_seconds": pytest.approx(29 * 109.5)}
    }
    assert estimate["estimated_hours"] == pytest.approx(29 * 109.5 / 3600, abs=0.01)
