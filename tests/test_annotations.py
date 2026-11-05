"""Checking, unblinding and summarizing qualitative annotations (E18). Synthetic labels only."""

import pytest

from src.analysis.annotations import (
    ANNOTATED_COLUMNS,
    category_summary,
    check_annotations,
    disputed_summary,
    parse_category,
    read_table,
    report_markdown,
    scores_by_category,
    split_disputed,
    unblind,
    wilson_interval,
)
from src.analysis.qualitative import ANNOTATION_COLUMNS, annotation_template, blind, review_rows, review_sheet_markdown
from src.evaluation.scoring import write_csv

IDS = [f"clotho_eval_{i:04d}" for i in range(20)]
MODELS = ["gpt_audio_1_5", "gemini_3_8_flash", "qwen2_5_omni", "audio_flamingo_3"]
CODES = {1, 2, 3, 4, 5, 6}


def sample() -> tuple[list[dict], list[dict], list[dict]]:
    """Review rows, key rows and a completed template with synthetic labels (category = position in the key % 6 + 1)."""
    references = {sid: [f"ref {k} of {sid}" for k in range(5)] for sid in IDS}
    captions = {
        m: {sid: {"caption": f"caption {i} for {sid}", "normalization": "none"} for sid in IDS}
        for i, m in enumerate(MODELS)
    }
    review, key = review_rows(blind(IDS[:10], MODELS, 42), references, captions)
    filled = annotation_template(review)
    for index, row in enumerate(filled):
        row["category"] = str(index % 6 + 1)
        row["disputed_details"] = "metallic clank; distant thunder" if index % 6 + 1 in (3, 5) else ""
    return review, key, filled


def test_parse_category():
    assert parse_category(" 3 ", CODES) == 3 and parse_category("3.0", CODES) == 3
    assert parse_category("", CODES) is None and parse_category("7", CODES) is None
    assert parse_category("2.5", CODES) is None and parse_category("x", CODES) is None


def test_check_annotations_catches_edits_and_missing_labels(tmp_path):
    review, _, filled = sample()
    path = tmp_path / "filled.csv"
    write_csv(filled, path, ANNOTATION_COLUMNS)
    (tmp_path / "bom.csv").write_bytes(b"\xef\xbb\xbf" + path.read_bytes())  # spreadsheet export with a BOM
    assert check_annotations(read_table(tmp_path / "bom.csv"), review, CODES) == []
    broken = [dict(row) for row in filled]
    broken[0]["caption"] = "edited"
    broken[1]["category"] = ""
    errors = check_annotations(broken, review, CODES)
    assert len(errors) == 2 and "caption" in errors[0] and "category" in errors[1]
    assert "reordered" in check_annotations(filled[::-1], review, CODES)[0]
    assert "missing columns" in check_annotations([{"item_id": "x"}], review, CODES)[0]


def test_unblind_joins_the_key():
    _, key, filled = sample()
    rows = unblind(filled, key)
    assert all(tuple(row) == ANNOTATED_COLUMNS for row in rows)
    assert [r["model_key"] for r in rows] == [k["model_key"] for k in key]
    assert rows[0]["category"] == 1
    with pytest.raises(ValueError):
        unblind(filled, key[:-1])


def test_wilson_interval():
    low, high = wilson_interval(0, 50, 0.95)
    assert low == 0.0 and high == pytest.approx(0.0713, abs=1e-4)
    low, high = wilson_interval(25, 50, 0.95)
    assert low == pytest.approx(1 - high) and low < 0.5 < high


def test_summaries():
    _, key, filled = sample()
    rows = unblind(filled, key)
    summary = category_summary(rows, MODELS, [1, 2, 3, 4, 5, 6], 0.95)
    assert len(summary) == 24
    for model in MODELS:
        assert sum(r["count"] for r in summary if r["model_key"] == model) == 10
    scores = {(m, sid): {"cider_d": 0.5} for m in MODELS for sid in IDS}
    by_category = scores_by_category(rows, scores, ["cider_d"], [1, 2, 3, 4, 5, 6, 7])
    assert by_category[0]["mean"] == 0.5 and by_category[-1]["num_items"] == 0 and by_category[-1]["mean"] is None
    disputed = disputed_summary(rows, MODELS)
    assert sum(r["disputed_phrases"] for r in disputed) == 2 * sum(r["category"] in (3, 5) for r in rows)
    assert split_disputed(" a ;; b; ") == ["a", "b"]


def test_report_and_sheet_have_no_notes_or_unblinded_text():
    review, key, filled = sample()
    filled[0]["notes_other"] = "private note"
    rows = unblind(filled, key)
    categories = [{"code": c, "name": f"cat {c}", "description": ""} for c in range(1, 7)]
    scores = {(m, sid): {"cider_d": 0.5} for m in MODELS for sid in IDS}
    report = {
        "run_info": {"git_commit": "abc", "git_dirty": False, "timestamp_utc": "t"},
        "display_names": {m: m.upper() for m in MODELS},
        "categories": categories,
        "confidence": 0.95,
        "metrics": ["cider_d"],
        "metric_labels": {"cider_d": "CIDEr-D"},
        "num_items": 40,
        "num_recordings": 10,
        "items_per_model": 10,
        "key_verified": True,
        "inputs": {"a": {"path": "p", "sha256": "0"}},
        "summary": category_summary(rows, MODELS, list(range(1, 7)), 0.95),
        "disputed": disputed_summary(rows, MODELS),
        "scores_by_category": scores_by_category(rows, scores, ["cider_d"], list(range(1, 7))),
        "limitations": ["x"],
        "outputs": {"r": "r.md"},
    }
    text = report_markdown(report)
    assert "private note" not in text and "for clotho" not in text and "metallic" not in text
    assert "| 1. cat 1 |" in text
    sheet = review_sheet_markdown(review)
    assert sheet.count("## R") == 10 and "## R01" in sheet and not any(m in sheet for m in MODELS)
