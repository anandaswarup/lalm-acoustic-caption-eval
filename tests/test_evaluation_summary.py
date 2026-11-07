"""Model summaries, paired differences, the sensitivity check and the CLAP audio cache (E11). No Java or models."""

import math

import numpy as np
import pytest

from src.evaluation.scoring import ScoreSet
from src.evaluation.summary import (
    ALL_RECORDINGS,
    VALID_FOR_ALL,
    difference_rows,
    report_markdown,
    restrict,
    sensitivity_rows,
    summary_rows,
)
from src.metrics.clap import load_audio_cache, save_audio_cache
from src.metrics.registry import Score

IDS = [f"s{i}" for i in range(40)]
BOOT = (200, 0.95, 42)


def tables(score_sets: list[ScoreSet], sample: str) -> tuple[list[dict], list[dict]]:
    return summary_rows(score_sets, sample, *BOOT), difference_rows(score_sets, sample, *BOOT)


def make_sets(gaps: dict[str, float], seed: int = 0) -> list[ScoreSet]:
    """One score set per system: shared per-item noise plus a fixed offset, for BLEU-1 and CIDEr-D."""
    noise = np.random.default_rng(seed).normal(0.5, 0.1, len(IDS))
    sets = []
    for system, gap in gaps.items():
        items = noise + gap
        scores = {"bleu_1": Score(0.9, items), "cider_d": Score(float(items.mean()), items * 2)}
        sets.append(ScoreSet(system, "five_reference", "all", list(IDS), scores))
    return sets


def test_restrict_keeps_order_and_drops_corpus_statistics():
    sets = make_sets({"a": 0.0, "b": 0.1})
    (a, _) = restrict(sets, ["s3", "s1"])
    assert a.sample_ids == ["s1", "s3"]
    assert math.isnan(a.scores["bleu_1"].corpus)  # corpus BLEU cannot be recomputed without rescoring
    assert a.scores["cider_d"].corpus == pytest.approx(a.scores["cider_d"].items.mean())
    np.testing.assert_array_equal(a.scores["cider_d"].items, sets[0].scores["cider_d"].items[[1, 3]])
    with pytest.raises(ValueError, match="not scored"):
        restrict(sets, ["s1", "x"])
    misaligned = [sets[0], ScoreSet("b", "five_reference", "all", IDS[::-1], sets[1].scores)]
    with pytest.raises(ValueError, match="same condition and recordings"):
        restrict(misaligned, ["s1"])


def test_summary_rows():
    rows = summary_rows(make_sets({"a": 0.0, "b": 0.1}), ALL_RECORDINGS, *BOOT)
    assert [(r["system"], r["metric"]) for r in rows] == [
        ("a", "bleu_1"), ("a", "cider_d"), ("b", "bleu_1"), ("b", "cider_d")
    ]  # fmt: skip
    row = rows[0]
    assert (row["sample"], row["num_items"], float(row["corpus"])) == (ALL_RECORDINGS, 40, 0.9)
    assert float(row["ci_low"]) < float(row["item_mean"]) < float(row["ci_high"])
    restricted = summary_rows(restrict(make_sets({"a": 0.0}), IDS[:10]), VALID_FOR_ALL, *BOOT)
    assert restricted[0]["corpus"] == "" and restricted[0]["num_items"] == 10
    # Same seed, same resamples: identical systems get identical intervals.
    same = summary_rows(make_sets({"a": 0.0, "b": 0.0}), ALL_RECORDINGS, *BOOT)
    assert same[0]["ci_low"] == same[2]["ci_low"] and same[0]["ci_high"] == same[2]["ci_high"]


def test_difference_rows_are_paired():
    rows = difference_rows(make_sets({"a": 0.0, "b": 0.1, "c": 0.1}), ALL_RECORDINGS, *BOOT)
    assert [(r["metric"], r["system_a"], r["system_b"]) for r in rows[:3]] == [
        ("bleu_1", "a", "b"), ("bleu_1", "a", "c"), ("bleu_1", "b", "c")
    ]  # fmt: skip
    ab, _, bc = rows[:3]
    # Every recording differs by exactly 0.1, so the paired interval has zero width.
    assert float(ab["difference"]) == pytest.approx(-0.1) and float(ab["ci_high"]) == pytest.approx(-0.1)
    assert ab["excludes_zero"] is True
    assert float(bc["difference"]) == 0.0 and bc["excludes_zero"] is False


def test_sensitivity_rows_flag_order_and_significance_changes():
    full_sets = make_sets({"a": 0.0, "b": 0.02})
    # On the sample, a is ahead of b and the difference flips sign.
    sample_sets = restrict(full_sets, IDS[:20])
    sample_sets[0].scores["bleu_1"] = Score(math.nan, sample_sets[0].scores["bleu_1"].items + 0.05)
    (full_s, full_d), (sample_s, sample_d) = tables(full_sets, ALL_RECORDINGS), tables(sample_sets, VALID_FOR_ALL)
    rows = {row["metric"]: row for row in sensitivity_rows(full_s, sample_s, full_d, sample_d)}
    bleu, cider = rows["bleu_1"], rows["cider_d"]
    assert (bleu["order_all"], bleu["order_sample"], bleu["order_changed"]) == ("b > a", "a > b", True)
    assert bleu["significance_changes"] == ""  # excludes 0 before and after, with opposite signs
    assert cider["order_changed"] is False and cider["significance_changes"] == ""
    shifts = [abs(after.scores["bleu_1"].items.mean() - before.scores["bleu_1"].items.mean())
              for before, after in zip(full_sets, sample_sets, strict=True)]  # fmt: skip
    assert float(bleu["max_abs_mean_change"]) == pytest.approx(max(shifts))
    # On another sample, a catches up exactly: the difference no longer excludes 0.
    sample_sets[0].scores["bleu_1"] = Score(math.nan, sample_sets[1].scores["bleu_1"].items.copy())
    sample_s, sample_d = tables(sample_sets, VALID_FOR_ALL)
    bleu = next(row for row in sensitivity_rows(full_s, sample_s, full_d, sample_d) if row["metric"] == "bleu_1")
    assert bleu["significance_changes"] == "a-b: excludes 0->includes 0"


def test_report_markdown_has_no_missing_cells():
    sets = make_sets({"a": 0.0, "b": 0.1})
    valid = restrict(sets, IDS[:30])
    summary = summary_rows(sets, ALL_RECORDINGS, *BOOT) + summary_rows(valid, VALID_FOR_ALL, *BOOT)
    differences = difference_rows(sets, ALL_RECORDINGS, *BOOT) + difference_rows(valid, VALID_FOR_ALL, *BOOT)
    pick = lambda rows, sample: [r for r in rows if r["sample"] == sample]  # noqa: E731
    sensitivity = sensitivity_rows(
        pick(summary, ALL_RECORDINGS), pick(summary, VALID_FOR_ALL),
        pick(differences, ALL_RECORDINGS), pick(differences, VALID_FOR_ALL),
    )  # fmt: skip
    report = {
        "run_info": {"git_commit": "abc", "git_dirty": False, "timestamp_utc": "now"},
        "task": "E11",
        "condition": "five_reference",
        "subset": "all",
        "smoke_test_limit": None,
        "display_names": {"a": "Model A", "b": "Model B"},
        "metrics": ["bleu_1", "cider_d"],
        "inputs": {
            "recordings": 40,
            "references_per_recording": 5,
            "captions": {"path": "c.jsonl", "sha256": "1"},
            "references": {"path": "r.json", "sha256": "2"},
            "normalization": {"a": {"none": 40}},
        },
        "bootstrap": {"num_resamples": 200, "confidence": 0.95, "seed": 42},
        "summary": summary,
        "differences": differences,
        "sensitivity": {"num_items": 30, "excluded": 10, "rows": sensitivity},
        "timing": {"seconds": {"bleu": 1.0}, "calls": {"bleu": 2}, "load_models_seconds": 3.0,
                   "scoring_seconds": 60.0, "clap_cache_hits": 0},
        "limitations": ["note"],
    }  # fmt: skip
    text = report_markdown(report)
    assert "| BLEU-1 | 0.9000 (" in text and "| CIDEr-D | " in text
    assert "Model A - Model B" in text and "-0.1000*" in text
    assert "30 recordings where every model's caption is valid" in text
    assert "nan" not in text


def test_clap_audio_cache_round_trip(tmp_path):
    path = tmp_path / "cache" / "clap_audio.npz"
    assert load_audio_cache(path, "k") == {}
    embeddings = {"b" * 64: np.arange(4.0) / 3, "a" * 64: np.ones(4)}
    save_audio_cache(path, "k", embeddings)
    loaded = load_audio_cache(path, "k")
    assert sorted(loaded) == sorted(embeddings)
    for sha, vector in embeddings.items():
        np.testing.assert_array_equal(loaded[sha], vector)  # bit-identical float64
    assert load_audio_cache(path, "other settings") == {}
    assert not path.with_name(path.name + ".tmp").exists()
