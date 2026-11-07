"""Blinded audio-grounded review of the qualitative sample (E18).

After the E17 labels are final, the reviewer listens to each sampled recording and rates every caption against the
audio, and checks every E17 disputed phrase. The review material keeps the E16 recording codes and caption labels and
shows each caption with its disputed phrases, never the references, E17 categories, model names, file names or scores.
Audio copies are named by recording code. Two templates: one row per caption (`audio_rating` and notes) and one row per
disputed phrase (`verdict` and a note).

Once both templates are complete they are checked against the review material, unblinded with the key and summarised:
rating and verdict shares per model with Wilson intervals (descriptive), the E17 category x audio rating cross-tab,
the category-6 split, and the mean five-reference scores per audio rating.
"""

from __future__ import annotations

import html
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.analysis.annotations import split_disputed, wilson_interval

ITEM_COLUMNS = ("item_id", "recording_code", "caption_label", "caption")
AUDIO_FIELDS = ("audio_rating", "notes_sources", "notes_attributes", "notes_temporal", "notes_other")
AUDIO_COLUMNS = (*ITEM_COLUMNS, *AUDIO_FIELDS)
PHRASE_REVIEW_COLUMNS = ("phrase_id", "item_id", "recording_code", "caption_label", "phrase")
PHRASE_FIELDS = ("verdict", "note")
PHRASE_COLUMNS = (*PHRASE_REVIEW_COLUMNS, *PHRASE_FIELDS)
WAV_CHUNKS = ("fmt ", "data")  # anything else (LIST/INFO, id3, bext) could carry a title or file name

POOLED = "all"  # model_key of the rows pooled over models
UNSUPPORTED = "U"
MISMATCH_CATEGORY, MAJOR_RATING = 6, 3  # E17 category 6 rated 3 is a real mismatch; rated 1-2, references incomplete
CAPTION_COLUMNS = (
    "item_id", "recording_code", "caption_label", "sample_id", "model_key", "normalization", "category",
    "audio_rating", "num_phrases", "num_unsupported", "verdicts", *AUDIO_FIELDS[1:],
)  # fmt: skip
PHRASE_ROW_COLUMNS = (
    "phrase_id", "item_id", "recording_code", "caption_label", "sample_id", "model_key", "category", "audio_rating",
    "phrase", "verdict", "note",
)  # fmt: skip
RATING_SUMMARY_COLUMNS = ("model_key", "audio_rating", "count", "total", "share", "ci_low", "ci_high")
VERDICT_SUMMARY_COLUMNS = ("model_key", "verdict", "count", "total", "share", "ci_low", "ci_high")
UNSUPPORTED_COLUMNS = (
    "model_key", "captions", "captions_with_phrases", "captions_with_unsupported", "share", "ci_low", "ci_high",
)  # fmt: skip
CROSSTAB_COLUMNS = ("model_key", "category", "audio_rating", "count")
MISMATCH_COLUMNS = ("model_key", "category_6", "real_mismatch", "references_incomplete")
SCORE_COLUMNS = ("audio_rating", "metric", "num_items", "mean", "sd")


def audio_template(review: list[dict[str, str]]) -> list[dict[str, str]]:
    """One blank row per caption, in review order."""
    blank = dict.fromkeys(AUDIO_FIELDS, "")
    return [{**{column: row[column] for column in ITEM_COLUMNS}, **blank} for row in review]


def phrase_template(annotations: list[dict[str, str]]) -> list[dict[str, str]]:
    """One blank row per E17 disputed phrase, in review order; phrase IDs are the item ID and the phrase's position
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
        "# Audio review sheet (E18)",
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
<title>Audio review (E18)</title>
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
<h1>Audio review (E18)</h1>
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


def parse_verdict(value: str, codes: set[str]) -> str | None:
    """The verdict code in a cell (" s " -> "S"), or None if it is empty or not a valid code."""
    text = value.strip().upper()
    return text if text in codes else None


def check_labels(
    rows: list[dict[str, str]],
    expected: list[dict[str, str]],
    fixed: tuple[str, ...],
    field: str,
    parse: Callable[[str], Any],
) -> list[str]:
    """Problems that stop the analysis: wrong columns, rows missing, added or reordered, fixed columns changed, and
    labels in `field` missing or invalid. `expected` is the blank template; the first fixed column is the row ID."""
    if not rows:
        return ["no rows"]
    if missing := [column for column in (*fixed, field) if column not in rows[0]]:
        return [f"missing columns {missing}"]
    id_column = fixed[0]
    if [row[id_column] for row in rows] != [row[id_column] for row in expected]:
        wanted, found = {r[id_column] for r in expected}, {r[id_column] for r in rows}
        return [
            f"{id_column}s differ from the template (missing {sorted(wanted - found)[:5]}, "
            f"extra {sorted(found - wanted)[:5]}, or reordered)"
        ]
    errors = []
    for row, original in zip(rows, expected, strict=True):
        if changed := [column for column in fixed if row[column].strip() != original[column].strip()]:
            errors.append(f"{row[id_column]}: fixed columns changed: {changed}")
        if parse(row[field]) is None:
            errors.append(f"{row[id_column]}: {field} {row[field]!r} is not valid")
    return errors


def unblind_audio(
    e18: list[dict[str, Any]], audio: list[dict[str, str]], phrases: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Checked audio and phrase labels joined with the unblinded E17 rows (sample, model, category), in review order.
    Returns one row per caption (with its phrase verdicts) and one row per phrase."""
    if [row["item_id"] for row in audio] != [row["item_id"] for row in e18]:
        raise ValueError("The audio template and the E17 annotations do not cover the same items in the same order")
    by_item = {row["item_id"]: row for row in e18}
    ratings = {row["item_id"]: int(float(row["audio_rating"])) for row in audio}
    verdicts: dict[str, list[str]] = {}
    phrase_rows = []
    for row in phrases:
        item = by_item[row["item_id"]]
        verdict = row["verdict"].strip().upper()
        verdicts.setdefault(row["item_id"], []).append(verdict)
        phrase_rows.append(
            {
                **{column: row[column] for column in PHRASE_REVIEW_COLUMNS[:4]},
                **{column: item[column] for column in ("sample_id", "model_key", "category")},
                "audio_rating": ratings[row["item_id"]],
                "phrase": row["phrase"],
                "verdict": verdict,
                "note": row["note"].strip(),
            }
        )
    captions = []
    for row in audio:
        item, found = by_item[row["item_id"]], verdicts.get(row["item_id"], [])
        captions.append(
            {
                **{column: item[column] for column in CAPTION_COLUMNS[:7]},
                "audio_rating": ratings[row["item_id"]],
                "num_phrases": len(found),
                "num_unsupported": found.count(UNSUPPORTED),
                "verdicts": ";".join(found),
                **{field: row[field].strip() for field in AUDIO_FIELDS[1:]},
            }
        )
    return captions, phrase_rows


def _groups(rows: list[dict[str, Any]], models: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Rows per model, then all rows under POOLED."""
    return {**{model: [row for row in rows if row["model_key"] == model] for model in models}, POOLED: rows}


def share_summary(
    rows: list[dict[str, Any]], models: list[str], field: str, codes: list, confidence: float
) -> list[dict[str, Any]]:
    """Count, share and Wilson interval of every code of `field` per model and pooled (zero counts included)."""
    out = []
    for model, items in _groups(rows, models).items():
        values = [row[field] for row in items]
        for code in codes:
            count = values.count(code)
            low, high = wilson_interval(count, len(values), confidence)
            out.append(
                {
                    "model_key": model,
                    field: code,
                    "count": count,
                    "total": len(values),
                    "share": count / len(values) if values else None,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return out


def unsupported_summary(captions: list[dict[str, Any]], models: list[str], confidence: float) -> list[dict[str, Any]]:
    """Captions with at least one unsupported phrase per model and pooled, as a share of all captions."""
    out = []
    for model, items in _groups(captions, models).items():
        count = sum(row["num_unsupported"] > 0 for row in items)
        low, high = wilson_interval(count, len(items), confidence)
        out.append(
            {
                "model_key": model,
                "captions": len(items),
                "captions_with_phrases": sum(row["num_phrases"] > 0 for row in items),
                "captions_with_unsupported": count,
                "share": count / len(items) if items else None,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return out


def crosstab(
    captions: list[dict[str, Any]], models: list[str], categories: list[int], ratings: list[int]
) -> list[dict[str, Any]]:
    """Caption counts per E17 category and audio rating, per model and pooled (zero cells included)."""
    return [
        {
            "model_key": model,
            "category": category,
            "audio_rating": rating,
            "count": sum(row["category"] == category and row["audio_rating"] == rating for row in items),
        }
        for model, items in _groups(captions, models).items()
        for category in categories
        for rating in ratings
    ]


def mismatch_split(captions: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    """E17 category-6 captions per model and pooled: rated 3 (real mismatch) or 1-2 (references incomplete)."""
    out = []
    for model, items in _groups(captions, models).items():
        weak = [row["audio_rating"] for row in items if row["category"] == MISMATCH_CATEGORY]
        major = weak.count(MAJOR_RATING)
        row = {"model_key": model, "category_6": len(weak), "real_mismatch": major}
        out.append({**row, "references_incomplete": len(weak) - major})
    return out


def rating_conflicts(captions: list[dict[str, Any]], accurate: int = 1) -> list[str]:
    """Items rated accurate that have an unsupported phrase, which the guidelines would rate 2 or 3. Reported only;
    labels are not changed after unblinding."""
    return [row["item_id"] for row in captions if row["audio_rating"] == accurate and row["num_unsupported"]]


AUDIO_LIMITATIONS = [
    "One reviewer rated every caption, so inter-annotator agreement cannot be estimated; more reviewers are left for "
    "future work.",
    "The reviewer had read the references and assigned the E17 categories before listening, so the audio judgements "
    "may be anchored on them; captions also carry model style cues (E16).",
    "Only the phrases E17 marked as unsupported by every reference were verified one by one; the verdict shares are "
    "therefore not error rates for all caption content.",
    "Ratings record what one listener hears in the recording; Clotho has no ground-truth source labels.",
    "With 50 captions per model, proportions are coarse (one caption is 2 percentage points) and the Wilson intervals "
    "wide. Pooled and phrase-level intervals treat the four captions of a recording and the phrases of a caption as "
    "independent, so they are too narrow; the summary is descriptive.",
]


def _cell(row: dict[str, Any]) -> str:
    return f"{row['count']} ({row['share']:.0%}) [{row['ci_low']:.0%}, {row['ci_high']:.0%}]"


def audio_report_markdown(report: dict[str, Any]) -> str:
    """The audio-review report: counts, shares and settings only, never caption text or notes."""
    names, run_info = report["display_names"], report["run_info"]
    models, scopes = list(names), [*names, POOLED]
    label = {**names, POOLED: "All"}
    ratings, verdicts = report["ratings"], report["verdicts"]
    head = "| " + " | ".join(label[m] for m in scopes) + " |"
    rule = "|---|" + "---|" * len(scopes)
    lines = [
        "# Audio-grounded review (E18)",
        "",
        f"Generated by `scripts/analyze_audio_review.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        "## Inputs",
        "",
        *[f"- `{item['path']}` (sha256 `{item['sha256']}`)" for item in report["inputs"].values()],
        f"- Blinding key SHA-256 matches `sample_manifest.json`, and the review inputs match "
        f"`audio_review_manifest.json`: {report['inputs_verified']}.",
        f"- {report['num_items']} captions ({report['num_recordings']} recordings x {len(models)} models) and "
        f"{report['num_phrases']} disputed phrases, one reviewer, guidelines in "
        "`outputs/qualitative/audio_review_guidelines.md`.",
        "",
        "## Audio ratings by model",
        "",
        f"Count and share of the model's {report['items_per_model']} captions, with Wilson "
        f"{report['confidence']:.0%} intervals.",
        "",
        "| Rating " + head,
        rule,
    ]
    summary = {(r["model_key"], r["audio_rating"]): r for r in report["rating_summary"]}
    for rating in ratings:
        code = rating["code"]
        lines.append(f"| {code}. {rating['name']} | " + " | ".join(_cell(summary[(m, code)]) for m in scopes) + " |")
    summary = {(r["model_key"], r["verdict"]): r for r in report["verdict_summary"]}
    lines += [
        "",
        "## Disputed phrases against the audio",
        "",
        "Verdicts on the phrases that E17 found in no reference, as a share of the model's disputed phrases.",
        "",
        "| Verdict " + head,
        rule,
        *[
            f"| {v['code']}. {v['name']} | " + " | ".join(_cell(summary[(m, v["code"])]) for m in scopes) + " |"
            for v in verdicts
        ],
        "| Phrases | " + " | ".join(str(summary[(m, verdicts[0]["code"])]["total"]) for m in scopes) + " |",
        "",
        "Captions with at least one unsupported phrase, as a share of all the model's captions:",
        "",
        "| Model | Captions with disputed phrases | With an unsupported phrase |",
        "|---|---|---|",
        *[
            f"| {label[r['model_key']]} | {r['captions_with_phrases']} of {r['captions']} | "
            f"{r['captions_with_unsupported']} ({r['share']:.0%}) [{r['ci_low']:.0%}, {r['ci_high']:.0%}] |"
            for r in report["unsupported_summary"]
        ],
        "",
        "## E17 category against audio rating",
        "",
        "Caption counts; rows are the reference-only E17 categories, columns the audio ratings.",
        "",
    ]
    cells = {(r["model_key"], r["category"], r["audio_rating"]): r["count"] for r in report["crosstab"]}
    codes = [r["code"] for r in ratings]
    for model in [POOLED, *models]:
        lines += [
            f"**{label[model]}**",
            "",
            "| Category | " + " | ".join(str(c) for c in codes) + " | Total |",
            "|---|" + "---|" * (len(codes) + 1),
        ]
        for category in report["categories"]:
            counts = [cells[(model, category["code"], c)] for c in codes]
            name = f"{category['code']}. {category['name']}"
            lines.append(f"| {name} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
        lines.append("")
    lines += [
        "## Category 6: real mismatches and incomplete references",
        "",
        f"E17 category-6 captions rated {MAJOR_RATING} against the audio are real mismatches; those rated 1 or 2 "
        "describe the recording correctly enough and point to incomplete references.",
        "",
        "| Model | Category 6 | Real mismatch (rated 3) | References incomplete (rated 1-2) |",
        "|---|---|---|---|",
        *[
            f"| {label[r['model_key']]} | {r['category_6']} | {r['real_mismatch']} | {r['references_incomplete']} |"
            for r in report["mismatch_split"]
        ],
        "",
        "## Consistency check",
        "",
        "Captions rated 1 (accurate) that have an unsupported phrase; the guidelines would rate them 2 or 3. Reported "
        f"only; labels are not changed after unblinding: {len(report['rating_conflicts'])}"
        + (f" ({', '.join(report['rating_conflicts'])})." if report["rating_conflicts"] else "."),
        "",
        "## Five-reference scores by audio rating",
        "",
        "Mean item score (E11) of the captions with each audio rating, pooled over models; descriptive only.",
        "",
        "| Rating | Items | " + " | ".join(report["metric_labels"][m] for m in report["metrics"]) + " |",
        "|---|---|" + "---|" * len(report["metrics"]),
    ]
    by = {(r["audio_rating"], r["metric"]): r for r in report["scores_by_audio_rating"]}
    for code in codes:
        count = by[(code, report["metrics"][0])]["num_items"]
        means = [f"{by[(code, m)]['mean']:.3f}" if count else "n/a" for m in report["metrics"]]
        lines.append(f"| {code} | {count} | " + " | ".join(means) + " |")
    lines += ["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""]
    lines += ["## Outputs", "", *[f"- `{path}`" for path in report["outputs"].values()], ""]
    return "\n".join(lines)
