"""Blinded audio-grounded review of the qualitative sample (E19).

After the E18 labels are final, the reviewer listens to each sampled recording and rates every caption against the
audio, and checks every E18 disputed phrase. The review material keeps the E17 recording codes and caption labels and
shows each caption with its disputed phrases, never the references, E18 categories, model names, file names or scores.
Audio copies are named by recording code. Two templates: one row per caption (`audio_rating` and notes) and one row per
disputed phrase (`verdict` and a note).
"""

from __future__ import annotations

import html
import struct
from pathlib import Path

from src.analysis.annotations import split_disputed

ITEM_COLUMNS = ("item_id", "recording_code", "caption_label", "caption")
AUDIO_FIELDS = ("audio_rating", "notes_sources", "notes_attributes", "notes_temporal", "notes_other")
AUDIO_COLUMNS = (*ITEM_COLUMNS, *AUDIO_FIELDS)
PHRASE_REVIEW_COLUMNS = ("phrase_id", "item_id", "recording_code", "caption_label", "phrase")
PHRASE_FIELDS = ("verdict", "note")
PHRASE_COLUMNS = (*PHRASE_REVIEW_COLUMNS, *PHRASE_FIELDS)
WAV_CHUNKS = ("fmt ", "data")  # anything else (LIST/INFO, id3, bext) could carry a title or file name


def audio_template(review: list[dict[str, str]]) -> list[dict[str, str]]:
    """One blank row per caption, in review order."""
    blank = dict.fromkeys(AUDIO_FIELDS, "")
    return [{**{column: row[column] for column in ITEM_COLUMNS}, **blank} for row in review]


def phrase_template(annotations: list[dict[str, str]]) -> list[dict[str, str]]:
    """One blank row per E18 disputed phrase, in review order; phrase IDs are the item ID and the phrase's position
    (R01B-1, R01B-2, ...)."""
    rows = []
    for row in annotations:
        for index, phrase in enumerate(split_disputed(row["disputed_details"]), start=1):
            rows.append(
                {
                    "phrase_id": f"{row['item_id']}-{index}",
                    **{column: row[column] for column in PHRASE_REVIEW_COLUMNS[1:4]},
                    "phrase": phrase,
                    **dict.fromkeys(PHRASE_FIELDS, ""),
                }
            )
    return rows


def has_labels(rows: list[dict[str, str]], fields: tuple[str, ...]) -> bool:
    """Whether any review field of a template is filled in (such a template is never overwritten)."""
    return any(row.get(field, "").strip() for row in rows for field in fields)


def wav_chunks(path: str | Path) -> list[str]:
    """Top-level RIFF chunk IDs of a WAV file."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    chunks, position = [], 12
    while position + 8 <= len(data):
        size = struct.unpack("<I", data[position + 4 : position + 8])[0]
        chunks.append(data[position : position + 4].decode("latin-1"))
        position += 8 + size + (size & 1)
    return chunks


def _by_code(review: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_code: dict[str, list[dict[str, str]]] = {}
    for row in review:
        by_code.setdefault(row["recording_code"], []).append(row)
    return by_code


def _phrases_by_item(phrases: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_item: dict[str, list[dict[str, str]]] = {}
    for row in phrases:
        by_item.setdefault(row["item_id"], []).append(row)
    return by_item


def review_sheet_markdown(review: list[dict[str, str]], phrases: list[dict[str, str]]) -> str:
    """Text of the audio review (no audio): per recording code, captions A-D and their disputed phrases."""
    by_item = _phrases_by_item(phrases)
    lines = [
        "# Audio review sheet (E19)",
        "",
        "Text of the audio review. Listen in `audio_review.html` (local only), follow `audio_review_guidelines.md`, "
        "and enter labels in `audio_annotations_reviewer_1.csv` and `disputed_verification_reviewer_1.csv`.",
        "",
    ]
    for code, rows in _by_code(review).items():
        lines += [f"## {code}", "", f"Audio: `audio/{code}.wav`", ""]
        for row in rows:
            lines.append(f"- **{row['item_id']}**: {row['caption']}")
            lines += [f"  - `{p['phrase_id']}` {p['phrase']}" for p in by_item.get(row["item_id"], [])]
        lines.append("")
    return "\n".join(lines)


def review_page_html(review: list[dict[str, str]], phrases: list[dict[str, str]], audio_dir: str) -> str:
    """Local listening page: per recording code, an audio player (`<audio_dir>/<code>.wav`) and captions A-D with
    their disputed phrases. Never committed (it links the audio copies)."""
    by_item = _phrases_by_item(phrases)
    sections = []
    for code, rows in _by_code(review).items():
        items = []
        for row in rows:
            listed = "".join(
                f"<li><code>{html.escape(p['phrase_id'])}</code> {html.escape(p['phrase'])}</li>"
                for p in by_item.get(row["item_id"], [])
            )
            disputed = f"<ul class='phrases'>{listed}</ul>" if listed else "<p class='none'>No disputed phrases</p>"
            caption = f"<strong>{html.escape(row['item_id'])}</strong> {html.escape(row['caption'])}"
            items.append(f"<li><p>{caption}</p>{disputed}</li>")
        sections.append(
            f"<section id='{code}'><h2>{code}</h2>"
            f"<audio controls preload='none' src='{audio_dir}/{code}.wav'></audio>"
            f"<ol class='captions'>{''.join(items)}</ol></section>"
        )
    nav = " ".join(f"<a href='#{code}'>{code}</a>" for code in _by_code(review))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio review (E19)</title>
<style>
:root {{ color-scheme: light dark; --fg: #1d1d1f; --bg: #fafaf8; --muted: #6b6b6b; --line: #ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg: #e8e8e6; --bg: #18181a; --muted: #9a9a9a; --line: #333; }} }}
body {{ font: 16px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg); max-width: 50rem;
  margin: 0 auto; padding: 1rem; }}
nav {{ font-size: 0.85rem; line-height: 1.9; }}
nav a {{ color: var(--muted); margin-right: 0.3rem; }}
section {{ border-top: 1px solid var(--line); padding: 1rem 0; }}
audio {{ width: 100%; }}
ol.captions {{ list-style: none; padding: 0; }}
ul.phrases, p.none {{ color: var(--muted); font-size: 0.9rem; margin-top: -0.5rem; }}
</style>
</head>
<body>
<h1>Audio review (E19)</h1>
<p>Listen to each recording in full, then rate every caption and check each disputed phrase, following
<code>audio_review_guidelines.md</code>. Enter labels in <code>audio_annotations_reviewer_1.csv</code> and
<code>disputed_verification_reviewer_1.csv</code>.</p>
<nav>{nav}</nav>
{"".join(sections)}
</body>
</html>
"""


def text_leaks(texts: dict[str, str], terms: list[str]) -> list[str]:
    """Named texts that contain a term (a model key or name, a file name or a reference), case-insensitive."""
    lowered = [(term, term.lower()) for term in terms if term]
    return [f"{name} contains {term!r}" for name, text in texts.items() for term, low in lowered if low in text.lower()]
