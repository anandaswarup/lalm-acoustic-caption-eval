"""Blinded audio-review material (E19). Synthetic captions, labels and WAV files only."""

import struct

import pytest

from src.analysis.audio_review import (
    AUDIO_COLUMNS,
    AUDIO_FIELDS,
    PHRASE_COLUMNS,
    PHRASE_FIELDS,
    audio_template,
    has_labels,
    phrase_template,
    review_page_html,
    review_sheet_markdown,
    text_leaks,
    wav_chunks,
)
from src.analysis.qualitative import annotation_template, blind, review_rows

IDS = [f"clotho_eval_{i:04d}" for i in range(20)]
MODELS = ["gpt_audio_1_5", "gemini_3_8_flash", "qwen2_5_omni", "audio_flamingo_3"]


def sample() -> tuple[list[dict], list[dict]]:
    """Review rows and an E18 template with synthetic disputed details on every third item."""
    references = {sid: [f"ref {k} of {sid}" for k in range(5)] for sid in IDS}
    captions = {
        m: {sid: {"caption": f"caption <{i}> & {sid}", "normalization": "none"} for sid in IDS}
        for i, m in enumerate(MODELS)
    }
    review, _ = review_rows(blind(IDS[:10], MODELS, 42), references, captions)
    filled = annotation_template(review)
    for index, row in enumerate(filled):
        row["category"] = "3"
        row["disputed_details"] = " metallic clank ;; distant thunder; " if index % 3 == 0 else ""
    return review, filled


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
