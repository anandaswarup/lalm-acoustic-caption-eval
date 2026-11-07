"""Blinded audio-review material (E19). Synthetic captions, labels and WAV files only."""

import struct

import pytest

from src.analysis.annotations import parse_category, scores_by_category, unblind
from src.analysis.audio_review import (
    AUDIO_COLUMNS,
    AUDIO_FIELDS,
    AUDIO_LIMITATIONS,
    CAPTION_COLUMNS,
    ITEM_COLUMNS,
    PHRASE_COLUMNS,
    PHRASE_FIELDS,
    PHRASE_REVIEW_COLUMNS,
    PHRASE_ROW_COLUMNS,
    POOLED,
    audio_report_markdown,
    audio_template,
    check_labels,
    crosstab,
    has_labels,
    mismatch_split,
    parse_verdict,
    phrase_template,
    rating_conflicts,
    review_page_html,
    review_sheet_markdown,
    share_summary,
    text_leaks,
    unblind_audio,
    unsupported_summary,
    wav_chunks,
)
from src.analysis.qualitative import annotation_template, blind, review_rows

IDS = [f"clotho_eval_{i:04d}" for i in range(20)]
MODELS = ["gpt_audio_1_5", "gemini_3_8_flash", "qwen2_5_omni", "audio_flamingo_3"]


def sample(with_key: bool = False) -> tuple[list[dict], ...]:
    """Review rows and an E18 template with synthetic disputed details on every third item (and the key)."""
    references = {sid: [f"ref {k} of {sid}" for k in range(5)] for sid in IDS}
    captions = {
        m: {sid: {"caption": f"caption <{i}> & {sid}", "normalization": "none"} for sid in IDS}
        for i, m in enumerate(MODELS)
    }
    review, key = review_rows(blind(IDS[:10], MODELS, 42), references, captions)
    filled = annotation_template(review)
    for index, row in enumerate(filled):
        row["category"] = "6" if index % 2 else "3"
        row["disputed_details"] = " metallic clank ;; distant thunder; " if index % 3 == 0 else ""
    return (review, filled, key) if with_key else (review, filled)


def labelled() -> tuple[list[dict], ...]:
    """Review rows, E18 annotations, key and completed E19 templates: ratings cycle 1, 2, 3 over the items; the first
    phrase of a caption is unsupported (lower case "u"), the second supported."""
    review, filled, key = sample(with_key=True)
    audio = audio_template(review)
    for index, row in enumerate(audio):
        row["audio_rating"] = str(index % 3 + 1)
        row["notes_other"] = "private note"
    phrases = phrase_template(filled)
    for row in phrases:
        row["verdict"] = " u " if row["phrase_id"].endswith("-1") else "S"
    return review, filled, key, audio, phrases


def wav(path, chunks: list[tuple[bytes, bytes]]):
    padded = [(cid, data + b"\0" * (len(data) & 1), len(data)) for cid, data in chunks]
    body = b"WAVE" + b"".join(cid + struct.pack("<I", size) + data for cid, data, size in padded)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def test_templates():
    review, filled = sample()
    audio = audio_template(review)
    assert len(audio) == 40 and all(tuple(row) == AUDIO_COLUMNS for row in audio)
    assert [row["item_id"] for row in audio] == [row["item_id"] for row in review]
    assert not any("reference" in column for column in AUDIO_COLUMNS)
    phrases = phrase_template(filled)
    assert len(phrases) == 2 * len(range(0, 40, 3)) and all(tuple(row) == PHRASE_COLUMNS for row in phrases)
    first = filled[0]["item_id"]
    expected = [(f"{first}-1", "metallic clank"), (f"{first}-2", "distant thunder")]
    assert [(p["phrase_id"], p["phrase"]) for p in phrases[:2]] == expected
    assert not has_labels(audio, AUDIO_FIELDS) and not has_labels(phrases, PHRASE_FIELDS)
    audio[3]["notes_other"] = " faint "
    phrases[1]["verdict"] = "S"
    assert has_labels(audio, AUDIO_FIELDS) and has_labels(phrases, PHRASE_FIELDS)


def test_sheet_and_page_are_blinded_and_escaped():
    review, filled = sample()
    phrases = phrase_template(filled)
    sheet = review_sheet_markdown(review, phrases)
    page = review_page_html(review, phrases, "audio")
    assert sheet.count("## R") == 10 and "`audio/R01.wav`" in sheet and "metallic clank" in sheet
    assert page.count("<audio ") == 10 and "src='audio/R10.wav'" in page
    assert "caption &lt;0&gt; &amp;" in page and "caption <0>" not in page
    assert text_leaks({"sheet": sheet, "page": page}, [*MODELS, "ref 0 of", "dog.wav"]) == []
    assert text_leaks({"x": "Recorded near DOG.WAV"}, ["dog.wav", ""]) == ["x contains 'dog.wav'"]


def test_wav_chunks(tmp_path):
    plain = wav(tmp_path / "a.wav", [(b"fmt ", b"\0" * 16), (b"data", b"\1\2\3")])
    assert wav_chunks(plain) == ["fmt ", "data"]
    tagged = wav(tmp_path / "b.wav", [(b"fmt ", b"\0" * 16), (b"LIST", b"INFOINAM"), (b"data", b"")])
    assert wav_chunks(tagged) == ["fmt ", "LIST", "data"]
    (tmp_path / "c.wav").write_bytes(b"not a wav file")
    with pytest.raises(ValueError):
        wav_chunks(tmp_path / "c.wav")


def test_parse_verdict_and_check_labels():
    assert parse_verdict(" s ", {"S", "U", "A"}) == "S" and parse_verdict("X", {"S"}) is None
    assert parse_verdict("", {"S"}) is None
    review, filled, _, audio, phrases = labelled()
    rating = lambda v: parse_category(v, {1, 2, 3})  # noqa: E731
    verdict = lambda v: parse_verdict(v, {"S", "U", "A"})  # noqa: E731
    assert check_labels(audio, audio_template(review), ITEM_COLUMNS, "audio_rating", rating) == []
    assert check_labels(phrases, phrase_template(filled), PHRASE_REVIEW_COLUMNS, "verdict", verdict) == []
    audio[0]["caption"] += " edited"
    audio[1]["audio_rating"] = "4"
    audio[2]["audio_rating"] = ""
    errors = check_labels(audio, audio_template(review), ITEM_COLUMNS, "audio_rating", rating)
    assert len(errors) == 3 and "fixed columns changed: ['caption']" in errors[0]
    assert "reordered" in check_labels(audio[::-1], audio_template(review), ITEM_COLUMNS, "audio_rating", rating)[0]
    phrases[0]["verdict"] = "maybe"
    assert len(check_labels(phrases, phrase_template(filled), PHRASE_REVIEW_COLUMNS, "verdict", verdict)) == 1
    missing = [{k: v for k, v in row.items() if k != "verdict"} for row in phrases]
    assert "missing columns" in check_labels(missing, phrases, PHRASE_REVIEW_COLUMNS, "verdict", verdict)[0]


def test_unblind_audio_and_summaries():
    _, filled, key, audio, phrases = labelled()
    captions, phrase_rows = unblind_audio(unblind(filled, key), audio, phrases)
    assert all(tuple(row) == CAPTION_COLUMNS for row in captions)
    assert all(tuple(row) == PHRASE_ROW_COLUMNS for row in phrase_rows)
    first = captions[0]
    assert first["audio_rating"] == 1 and first["num_phrases"] == 2 and first["verdicts"] == "U;S"
    assert first["num_unsupported"] == 1 and first["model_key"] == key[0]["model_key"]
    assert phrase_rows[0]["verdict"] == "U" and phrase_rows[0]["audio_rating"] == 1
    with pytest.raises(ValueError):
        unblind_audio(unblind(filled, key), audio[1:], phrases)

    ratings = share_summary(captions, MODELS, "audio_rating", [1, 2, 3], 0.95)
    assert len(ratings) == 3 * (len(MODELS) + 1)
    pooled = {r["audio_rating"]: r for r in ratings if r["model_key"] == POOLED}
    assert [pooled[c]["count"] for c in (1, 2, 3)] == [14, 13, 13] and pooled[1]["total"] == 40
    assert sum(r["count"] for r in ratings if r["model_key"] == MODELS[0]) == 10
    verdicts = {r["verdict"]: r for r in share_summary(phrase_rows, MODELS, "verdict", ["S", "U", "A"], 0.95)
                if r["model_key"] == POOLED}  # fmt: skip
    assert (verdicts["S"]["count"], verdicts["U"]["count"], verdicts["A"]["count"]) == (14, 14, 0)
    unsupported = {r["model_key"]: r for r in unsupported_summary(captions, MODELS, 0.95)}
    assert unsupported[POOLED]["captions_with_unsupported"] == 14 == unsupported[POOLED]["captions_with_phrases"]

    table = crosstab(captions, MODELS, [1, 2, 3, 4, 5, 6], [1, 2, 3])
    assert len(table) == (len(MODELS) + 1) * 18
    assert sum(r["count"] for r in table if r["model_key"] == POOLED) == 40
    split = {r["model_key"]: r for r in mismatch_split(captions, MODELS)}
    weak = [row["audio_rating"] for row in captions if row["category"] == 6]
    assert split[POOLED]["category_6"] == 20 == len(weak)
    assert split[POOLED]["real_mismatch"] == weak.count(3)
    assert split[POOLED]["references_incomplete"] == 20 - weak.count(3)
    assert rating_conflicts(captions) == [r["item_id"] for r in captions if r["audio_rating"] == 1 and r["num_phrases"]]


def test_audio_report_has_no_caption_text_or_notes():
    _, filled, key, audio, phrases = labelled()
    captions, phrase_rows = unblind_audio(unblind(filled, key), audio, phrases)
    scores = {(row["model_key"], row["sample_id"]): {"cider_d": 0.5} for row in captions}
    by_rating = scores_by_category(captions, scores, ["cider_d"], [1, 2, 3], field="audio_rating")
    assert [r["audio_rating"] for r in by_rating] == [1, 2, 3] and by_rating[0]["mean"] == 0.5
    categories = [{"code": c, "name": f"Category {c}"} for c in range(1, 7)]
    report = {
        "run_info": {"git_commit": "abc", "git_dirty": False, "timestamp_utc": "now"},
        "display_names": {m: m.upper() for m in MODELS},
        "categories": categories,
        "ratings": [{"code": c, "name": f"Rating {c}"} for c in (1, 2, 3)],
        "verdicts": [{"code": c, "name": f"Verdict {c}"} for c in ("S", "U", "A")],
        "confidence": 0.95,
        "metrics": ["cider_d"],
        "metric_labels": {"cider_d": "CIDEr-D"},
        "num_items": 40,
        "num_recordings": 10,
        "num_phrases": len(phrase_rows),
        "items_per_model": 10,
        "inputs_verified": True,
        "inputs": {"audio": {"path": "a.csv", "sha256": "0" * 64}},
        "rating_summary": share_summary(captions, MODELS, "audio_rating", [1, 2, 3], 0.95),
        "verdict_summary": share_summary(phrase_rows, MODELS, "verdict", ["S", "U", "A"], 0.95),
        "unsupported_summary": unsupported_summary(captions, MODELS, 0.95),
        "crosstab": crosstab(captions, MODELS, list(range(1, 7)), [1, 2, 3]),
        "mismatch_split": mismatch_split(captions, MODELS),
        "scores_by_audio_rating": by_rating,
        "rating_conflicts": rating_conflicts(captions),
        "limitations": AUDIO_LIMITATIONS,
        "outputs": {"report_md": "r.md"},
    }
    text = audio_report_markdown(report)
    assert "| 1. Rating 1 |" in text and "| Phrases |" in text and "**All**" in text
    assert "caption <" not in text and "private note" not in text and "metallic clank" not in text
