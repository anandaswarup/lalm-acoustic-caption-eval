"""Representative caption comparisons and qualitative figures (E19). Synthetic labels only."""

import pytest

from src.analysis.examples import (
    PATTERNS,
    Pattern,
    by_recording,
    caption_records,
    eligible,
    examples_latex,
    examples_markdown,
    latex_escape,
    recording_example,
    select_examples,
)
from src.analysis.qualitative_figures import CATEGORY_COLORS, RATING_COLORS, heatmap, save, stacked_bars

MODELS = ["m1", "m2", "m3", "m4"]


def record(code: str, model: str, category: int, rating: int, verdicts: str = "") -> dict:
    phrases = [(f"phrase {i}", v) for i, v in enumerate(verdicts)]
    return {
        "item_id": f"{code}{'ABCD'[MODELS.index(model)]}",
        "recording_code": code,
        "sample_id": f"clotho_eval_{code}",
        "model_key": model,
        "references": [f"ref {k} & 50%" for k in range(1, 6)],
        "caption": f"caption_{code}_{model}",
        "category": category,
        "audio_rating": rating,
        "phrases": phrases,
        "verdicts": [v for _, v in phrases],
    }


def recordings(spec: dict[str, list[tuple]]) -> list[dict]:
    return [
        record(code, model, *labels) for code, rows in spec.items() for model, labels in zip(MODELS, rows, strict=True)
    ]


ONE = Pattern("one", "Rated 1", "rating 1", lambda c: c["audio_rating"] == 1)
CAT6 = Pattern("cat6", "Category 6", "category 6", lambda c: c["category"] == 6)


def test_caption_records_joins_labels_and_phrases():
    review = [
        {"item_id": "R01A", "recording_code": "R01", "caption_label": "A", "caption": "a dog",
         **{f"reference_{k}": f"ref {k}" for k in range(1, 6)}},
    ]  # fmt: skip
    captions = [{"item_id": "R01A", "sample_id": "s1", "model_key": "m1", "category": "6", "audio_rating": "1"}]
    phrases = [{"item_id": "R01A", "phrase": "barking", "verdict": "U"}]
    [row] = caption_records(review, captions, phrases)
    assert row["category"] == 6 and row["audio_rating"] == 1 and row["references"][4] == "ref 5"
    assert row["phrases"] == [("barking", "U")] and row["verdicts"] == ["U"]
    with pytest.raises(ValueError):
        caption_records(review, [], phrases)


def test_selection_prefers_most_qualifying_and_unused_recordings():
    records = recordings({
        "R01": [(6, 1), (6, 1), (6, 1), (1, 2)],  # 3 rated 1, 3 cat 6
        "R02": [(1, 1), (1, 1), (2, 2), (2, 2)],  # 2 rated 1
        "R03": [(6, 3), (2, 2), (2, 2), (2, 2)],  # 1 cat 6
        "R04": [(2, 1), (2, 2), (2, 2), (2, 2)],  # 1 rated 1
    })  # fmt: skip
    assert eligible(records, CAT6) == {"R01": ["R01A", "R01B", "R01C"], "R03": ["R03A"]}
    # cat6 has fewer eligible recordings, so it chooses first and takes R01; "one" then prefers the unused R02.
    selection = select_examples(records, (ONE, CAT6), seed=42)
    assert [(s["pattern"], s["recording_code"]) for s in selection] == [("one", "R02"), ("cat6", "R01")]
    assert selection[0]["num_eligible_recordings"] == 3 and selection[0]["num_qualifying_captions"] == 6
    assert selection[1]["qualifying_items"] == ["R01A", "R01B", "R01C"]
    # With only one pattern, the recording with the most qualifying captions wins.
    assert select_examples(records, (ONE,), seed=42)[0]["recording_code"] == "R01"
    # A pattern whose only recording is taken reuses it rather than returning nothing.
    only = Pattern("only", "R01 only", "R01", lambda c: c["recording_code"] == "R01")
    also = Pattern("also", "R01 too", "R01", lambda c: c["recording_code"] == "R01")
    assert [s["recording_code"] for s in select_examples(records, (only, also), seed=1)] == ["R01", "R01"]
    assert [s["recording_code"] for s in select_examples(records, (CAT6, only), seed=1)] == ["R03", "R01"]
    never = Pattern("never", "Never", "none", lambda c: False)
    assert select_examples(records, (never,), seed=42)[0]["recording_code"] is None


def test_selection_ties_are_seeded_and_deterministic():
    records = recordings({f"R{i:02d}": [(1, 1), (2, 2), (2, 2), (2, 2)] for i in range(1, 21)})
    picks = {select_examples(records, (ONE,), seed=seed)[0]["recording_code"] for seed in range(10)}
    assert len(picks) > 1  # the seed decides among tied recordings
    assert select_examples(records, (ONE,), seed=3) == select_examples(records, (ONE,), seed=3)


def test_patterns():
    by_key = {p.key: p for p in PATTERNS}
    assert len(PATTERNS) == 6
    assert by_key["strong_agreement"].qualifies(record("R01", "m1", 2, 1))
    assert not by_key["strong_agreement"].qualifies(record("R01", "m1", 3, 1))
    assert by_key["supported_extra_detail"].qualifies(record("R01", "m1", 5, 2, "US"))
    assert not by_key["supported_extra_detail"].qualifies(record("R01", "m1", 4, 1, "S"))
    assert by_key["unsupported_detail"].qualifies(record("R01", "m1", 1, 1, "SU"))
    assert by_key["omission"].qualifies(record("R01", "m1", 4, 2))
    assert by_key["real_mismatch"].qualifies(record("R01", "m1", 6, 3))
    assert by_key["incomplete_references"].qualifies(record("R01", "m1", 6, 1))
    assert not by_key["incomplete_references"].qualifies(record("R01", "m1", 6, 2))


def test_tables():
    records = recordings({"R01": [(1, 1), (6, 3, "U"), (3, 2, "SA"), (4, 1)], "R02": [(1, 1)] * 4})
    names = {m: m.upper() for m in MODELS}
    files = {f"clotho_eval_{c}": f"{c}_x.wav" for c in ("R01", "R02")}
    examples = [recording_example(items, list(reversed(MODELS)), files) for items in by_recording(records).values()]
    assert [c["model_key"] for c in examples[0]["captions"]] == list(reversed(MODELS))
    text = examples_markdown("T", ["intro"], examples, names, {"R01": "Mismatch"})
    assert "## Mismatch: R01" in text and "## R02" in text and "`R01_x.wav`" in text
    assert "| M2 (`R01B`) | caption_R01_m2 | 6 | 3 | phrase 0 (U) |" in text
    latex = examples_latex(examples, names, {"R01": "First_one", "R02": "Second"}, "tab:x-", clearpage_every=1)
    assert latex.count("\\begin{table}") == 2 and latex.count("\\clearpage") == 1  # not after the last table
    assert "\\label{tab:x-r01}" in latex and "ref 1 \\& 50\\%" in latex
    assert "\\emph{Disputed: phrase 0 (S); phrase 1 (A)}" in latex
    assert "First\\_\\allowbreak{}one" in latex
    assert latex_escape("a_b & {c} ~ ^ \\ $ #") == (
        "a\\_\\allowbreak{}b \\& \\{c\\} \\textasciitilde{} \\textasciicircum{} \\textbackslash{} \\$ \\#"
    )


def test_figures_are_written_and_reproducible(tmp_path):
    counts = {"Model A": [3, 6, 6, 10, 10, 15], "Model B": [2, 2, 4, 6, 12, 24]}
    labels = [f"{c} label" for c in range(1, 7)]
    first = save(stacked_bars(counts, labels, CATEGORY_COLORS, "Title", "Captions"), tmp_path / "a", ["pdf", "png"], 72)
    again = save(stacked_bars(counts, labels, CATEGORY_COLORS, "Title", "Captions"), tmp_path / "b", ["pdf", "png"], 72)
    assert [p.name for p in first] == ["a.pdf", "a.png"]
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in again]
    fig = heatmap([[10, 3, 2], [0, 0, 0]], ["1 x", "2 y"], ["1", "2", "3"], "Title", "Rating", "Category")
    assert save(fig, tmp_path / "h", ["png"], 72)[0].stat().st_size > 0
    with pytest.raises(ValueError):
        stacked_bars({"M": [1, 2]}, ["a", "b"], RATING_COLORS, "T", "C")  # three colours for two codes
