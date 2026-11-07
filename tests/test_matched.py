"""Score-table round trip, human baseline, model-human differences and the matched report (E13). No Java or models."""

import json
import math

import numpy as np
import pytest

from src.evaluation.matched import HUMAN, load_human_baseline, matched_tables, report_markdown, verdict, verdict_rows
from src.evaluation.scoring import (
    CORPUS_COLUMNS,
    SCORE_COLUMNS,
    ScoreSet,
    corpus_rows,
    read_score_sets,
    score_rows,
    write_csv,
)
from src.evaluation.summary import (
    ALL_RECORDINGS,
    POSITION_MEAN,
    VALID_FOR_ALL,
    average_subsets,
    matched_difference_rows,
    summary_rows,
)
from src.metrics.registry import Score

IDS = [f"s{i}" for i in range(30)]
SUBSETS = [f"drop_{k}" for k in range(1, 6)]
BOOT = (200, 0.95, 42)


def subset_sets(system: str, condition: str, offset: float, seed: int = 0) -> list[ScoreSet]:
    """Per-subset score sets: shared per-item noise plus a system offset and a small subset effect."""
    noise = np.random.default_rng(seed).normal(0.5, 0.1, len(IDS))
    sets = []
    for k, subset in enumerate(SUBSETS):
        items = noise + offset + 0.01 * k
        scores = {"bleu_1": Score(0.4 + offset + 0.01 * k, items), "cider_d": Score(float(items.mean()), items * 2)}
        sets.append(ScoreSet(system, condition, subset, list(IDS), scores))
    return sets


def write_tables(tmp_path, sets: list[ScoreSet], suffix: str = ".csv"):
    scores = write_csv([r for s in sets for r in score_rows(s)], tmp_path / f"scores{suffix}", SCORE_COLUMNS)
    corpus = write_csv([r for s in sets for r in corpus_rows(s)], tmp_path / "corpus.csv", CORPUS_COLUMNS)
    return scores, corpus


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_read_score_sets_round_trip(tmp_path, suffix):
    sets = subset_sets("gpt", "four_reference", 0.0) + subset_sets("af3", "four_reference", 0.1)
    averaged = average_subsets(sets[:5])
    scores, corpus = write_tables(tmp_path, sets, suffix)
    write_csv([r for s in [*sets, averaged] for r in corpus_rows(s)], corpus, CORPUS_COLUMNS)  # extra rows ignored
    back = read_score_sets(scores, corpus)
    assert [(s.system, s.subset) for s in back] == [(s.system, s.subset) for s in sets]
    for before, after in zip(sets, back, strict=True):
        assert after.sample_ids == IDS and list(after.scores) == ["bleu_1", "cider_d"]
        for name, score in before.scores.items():
            assert after.scores[name].corpus == score.corpus
            np.testing.assert_array_equal(after.scores[name].items, score.items)  # exact, not approximate


def test_gzip_tables_are_reproducible(tmp_path):
    sets = subset_sets("gpt", "four_reference", 0.0)
    rows = [r for s in sets for r in score_rows(s)]
    first = write_csv(rows, tmp_path / "a" / "scores.csv.gz", SCORE_COLUMNS).read_bytes()
    second = write_csv(rows, tmp_path / "b" / "scores.csv.gz", SCORE_COLUMNS).read_bytes()
    assert first == second and first[:2] == b"\x1f\x8b"


def test_read_score_sets_rejects_bad_tables(tmp_path):
    sets = subset_sets("gpt", "four_reference", 0.0)[:1]
    rows = [r for s in sets for r in score_rows(s)]
    corpus = write_csv([r for s in sets for r in corpus_rows(s)], tmp_path / "corpus.csv", CORPUS_COLUMNS)
    duplicate = write_csv(rows + rows[:1], tmp_path / "dup.csv", SCORE_COLUMNS)
    with pytest.raises(ValueError, match="duplicate"):
        read_score_sets(duplicate, corpus)
    unknown = write_csv([{**rows[0], "metric": "nope"}], tmp_path / "unknown.csv", SCORE_COLUMNS)
    with pytest.raises(ValueError, match="unknown metrics"):
        read_score_sets(unknown, corpus)
    no_corpus = write_csv(rows, tmp_path / "scores.csv", SCORE_COLUMNS)
    with pytest.raises(ValueError, match="no corpus value"):
        read_score_sets(no_corpus, write_csv([], tmp_path / "empty.csv", CORPUS_COLUMNS))


def human_files(tmp_path, sets: list[ScoreSet]):
    """E12-like tables and report for `sets`."""
    scores, corpus = write_tables(tmp_path, sets)
    averaged = average_subsets(sets)
    report = {
        "condition": "human_loo",
        "smoke_test_limit": None,
        "subsets": SUBSETS,
        "metrics": list(sets[0].scores),
        "inputs": {"recordings": len(IDS)},
        "summary": summary_rows([averaged], ALL_RECORDINGS, *BOOT),
    }
    path = tmp_path / "human_loo_evaluation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return scores, corpus, path


def test_load_human_baseline(tmp_path):
    sets = subset_sets(HUMAN, "human_loo", 0.2)
    scores, corpus, report = human_files(tmp_path, sets)
    loaded = load_human_baseline(scores, corpus, report, SUBSETS, IDS)
    assert [s.subset for s in loaded] == SUBSETS and loaded[0].sample_ids == IDS
    smoke = load_human_baseline(scores, corpus, report, SUBSETS, IDS[:5])
    assert smoke[0].sample_ids == IDS[:5] and math.isnan(smoke[0].scores["bleu_1"].corpus)
    with pytest.raises(ValueError, match="subsets"):
        load_human_baseline(scores, corpus, report, SUBSETS[:4], IDS)
    # A changed score no longer reproduces the report's baseline.
    sets[2].scores["cider_d"].items[0] += 1e-12
    write_tables(tmp_path, sets)
    with pytest.raises(ValueError, match="does not reproduce"):
        load_human_baseline(scores, corpus, report, SUBSETS, IDS)


def test_matched_difference_rows_and_verdicts():
    models = [
        average_subsets(subset_sets(m, "four_reference", gap)) for m, gap in [("a", -0.1), ("b", 0.0), ("c", 0.1)]
    ]
    human = average_subsets(subset_sets(HUMAN, "human_loo", 0.0))
    rows = matched_difference_rows(models, human, ALL_RECORDINGS, *BOOT)
    assert [(r["metric"], r["system_a"], r["system_b"]) for r in rows[:3]] == [
        ("bleu_1", "a", HUMAN), ("bleu_1", "b", HUMAN), ("bleu_1", "c", HUMAN)
    ]  # fmt: skip
    assert rows[0]["condition"] == "four_reference" and float(rows[0]["difference"]) == pytest.approx(-0.1)
    assert [verdict(r) for r in rows[:3]] == ["lower", "not_different", "higher"]
    verdicts = verdict_rows(rows)
    assert verdicts[0] == {
        "sample": ALL_RECORDINGS, "metric": "bleu_1", "higher": "c", "not_different": "b", "lower": "a",
        "uniform": False,
    }  # fmt: skip
    below = matched_difference_rows(models[:1], human, ALL_RECORDINGS, *BOOT)
    assert all(r["uniform"] for r in verdict_rows(below))
    with pytest.raises(ValueError, match="same subset and recordings"):
        matched_difference_rows(models, subset_sets(HUMAN, "human_loo", 0.0)[0], ALL_RECORDINGS, *BOOT)
    other = ScoreSet(HUMAN, "human_loo", POSITION_MEAN, IDS, {"bleu_1": human.scores["bleu_1"]})
    with pytest.raises(ValueError, match="different metrics"):
        matched_difference_rows(models, other, ALL_RECORDINGS, *BOOT)


def matched_inputs():
    score_sets = subset_sets("a", "four_reference", -0.1) + subset_sets("b", "four_reference", 0.05)
    human_sets = subset_sets(HUMAN, "human_loo", 0.0)
    overlaps = {sid: [0.9 if (i + k) % 4 == 0 else 0.3 for k in range(5)] for i, sid in enumerate(IDS)}
    return score_sets, human_sets, overlaps


def test_matched_tables():
    score_sets, human_sets, overlaps = matched_inputs()
    sections = matched_tables(score_sets, human_sets, IDS[:20], overlaps, [0.6, 0.8], BOOT)
    samples = [ALL_RECORDINGS, VALID_FOR_ALL, "closest_overlap_below_0.6", "closest_overlap_below_0.8"]
    assert [s["sample"] for s in sections["sensitivity"]["samples"]] == samples[1:]
    assert [s["num_items"] for s in sections["sensitivity"]["samples"]] == [20, 30, 30]
    summary = sections["summary"]
    assert {(r["system"], r["subset"]) for r in summary if r["sample"] == ALL_RECORDINGS} == {
        *((m, s) for m in "ab" for s in SUBSETS), ("a", POSITION_MEAN), ("b", POSITION_MEAN), (HUMAN, POSITION_MEAN)
    }  # fmt: skip
    assert {r["sample"] for r in sections["matched"]} == set(samples)
    assert len(sections["differences"]) == len(samples) * 2  # one model pair, two metrics
    # Model and human samples are masked alike: the near-paraphrase sample drops the same (recording, subset) pairs.
    masked = [r for r in summary if r["sample"] == "closest_overlap_below_0.6" and r["metric"] == "cider_d"]
    assert len({r["num_items"] for r in masked}) == 1 and all(r["subset"] == POSITION_MEAN for r in masked)
    rows = sections["sensitivity"]["rows"]
    assert len(rows) == 3 * 2 and rows[0]["order_all"] == "b > human > a"
    with pytest.raises(ValueError, match="human subsets"):
        matched_tables(score_sets[:4] + score_sets[5:], human_sets, IDS, overlaps, [0.6], BOOT)


def test_report_markdown_has_no_missing_cells():
    score_sets, human_sets, overlaps = matched_inputs()
    sections = matched_tables(score_sets, human_sets, IDS[:20], overlaps, [0.6, 0.8], BOOT)
    report = {
        "run_info": {"git_commit": "abc", "git_dirty": False, "timestamp_utc": "now"},
        "condition": "four_reference",
        "subsets": SUBSETS,
        "smoke_test_limit": 30,
        "display_names": {"a": "Model A", "b": "Model B"},
        "metrics": ["bleu_1", "cider_d"],
        "inputs": {
            "recordings": 30, "references_per_subset": 4,
            "captions": {"path": "c.jsonl", "sha256": "1"}, "references": {"path": "r.json", "sha256": "2"},
            "human_baseline": {"scores": {"path": "h.csv", "sha256": "3"}, "corpus": {"path": "hc.csv", "sha256": "4"},
                               "report": "h.json", "git_commit": "def"},
        },
        "bootstrap": {"num_resamples": 200, "confidence": 0.95, "seed": 42},
        **sections,
        "timing": {"seconds": {"bleu": 1.0}, "calls": {"bleu": 10}, "load_models_seconds": 3.0,
                   "scoring_seconds": 60.0, "clap_cache_hits": 30},
        "limitations": ["note"],
    }  # fmt: skip
    text = report_markdown(report)
    assert "Smoke test on the first 30 recordings" in text
    assert "| Metric | Model A | Model B | Human |" in text
    assert "| CIDEr-D | -0.2000* | +0.1000* | above: Model B; below: Model A |" in text
    assert "models fall on different sides of the human baseline: **BLEU-1, CIDEr-D**" in text
    assert "Model A - Model B" in text and "### `closest_overlap_below_0.8`" in text
    assert "| BLEU-1 | 0.0400 | 0.0400 |" in text  # per-subset corpus range
    assert "nan" not in text
