"""Checking, unblinding and summarizing the qualitative annotations (E17).

One reviewer labels every item of the E16 sample with one agreement category (1-6), the disputed caption phrases and
optional notes. The completed template must keep every review column of `review_items.csv` unchanged; the blinding
key must match the SHA-256 recorded in the sample manifest. Category proportions per model get Wilson score intervals;
they are descriptive (50 items per model, one reviewer).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from src.analysis.qualitative import ANNOTATION_COLUMNS, ANNOTATION_FIELDS, KEY_COLUMNS, REVIEW_COLUMNS

ANNOTATED_COLUMNS = (
    "item_id", "recording_code", "caption_label", "sample_id", "model_key", "normalization", *ANNOTATION_FIELDS,
)  # fmt: skip
SUMMARY_COLUMNS = ("model_key", "category", "count", "total", "share", "ci_low", "ci_high")
SCORE_COLUMNS = ("category", "metric", "num_items", "mean", "sd")
DISPUTED_COLUMNS = ("model_key", "items_with_disputed_details", "disputed_phrases", "total")


def read_table(path: str | Path) -> list[dict[str, str]]:
    """A CSV saved by a spreadsheet: UTF-8 with or without a byte-order mark."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_category(value: str, codes: set[int]) -> int | None:
    """The category code in a cell ("3", " 3 ", "3.0"), or None if it is empty or not a valid code."""
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() and int(number) in codes else None


def check_annotations(annotations: list[dict[str, str]], review: list[dict[str, str]], codes: set[int]) -> list[str]:
    """Problems that stop the analysis: wrong columns, items missing, added or reordered, review columns changed, and
    categories missing or invalid."""
    if not annotations:
        return ["no annotation rows"]
    if missing := [column for column in ANNOTATION_COLUMNS if column not in annotations[0]]:
        return [f"missing columns {missing}"]
    errors = []
    if [row["item_id"] for row in annotations] != [row["item_id"] for row in review]:
        expected, found = {r["item_id"] for r in review}, {r["item_id"] for r in annotations}
        errors.append(
            f"item IDs differ from review_items.csv (missing {sorted(expected - found)[:5]}, "
            f"extra {sorted(found - expected)[:5]}, or reordered)"
        )
        return errors
    for row, original in zip(annotations, review, strict=True):
        changed = [column for column in REVIEW_COLUMNS if row[column].strip() != original[column].strip()]
        if changed:
            errors.append(f"{row['item_id']}: review columns changed: {changed}")
        if parse_category(row["category"], codes) is None:
            errors.append(f"{row['item_id']}: category {row['category']!r} is not one of {sorted(codes)}")
    return errors


def unblind(annotations: list[dict[str, str]], key: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Annotation rows joined with the key (sample ID, model, normalization), in review order."""
    by_item = {row["item_id"]: row for row in key}
    if set(by_item) != {row["item_id"] for row in annotations} or len(by_item) != len(key):
        raise ValueError("The blinding key does not cover exactly the annotated items")
    rows = []
    for row in annotations:
        entry = by_item[row["item_id"]]
        if (entry["recording_code"], entry["caption_label"]) != (row["recording_code"], row["caption_label"]):
            raise ValueError(f"{row['item_id']}: key and annotation disagree on recording code or label")
        rows.append(
            {
                **{column: entry[column] for column in KEY_COLUMNS},
                **{field: row[field].strip() for field in ANNOTATION_FIELDS},
                "category": int(float(row["category"])),
            }
        )
    return [{column: row[column] for column in ANNOTATED_COLUMNS} for row in rows]


def wilson_interval(count: int, total: int, confidence: float) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    share = count / total
    centre = (share + z**2 / (2 * total)) / (1 + z**2 / total)
    half = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def category_summary(
    rows: list[dict[str, Any]], models: list[str], codes: list[int], confidence: float
) -> list[dict[str, Any]]:
    """Count, share and Wilson interval of every category for every model (zero counts included)."""
    out = []
    for model in models:
        categories = [row["category"] for row in rows if row["model_key"] == model]
        for code in codes:
            count = categories.count(code)
            low, high = wilson_interval(count, len(categories), confidence)
            out.append(
                {
                    "model_key": model,
                    "category": code,
                    "count": count,
                    "total": len(categories),
                    "share": count / len(categories) if categories else math.nan,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return out


def scores_by_category(
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, float]],
    metrics: list[str],
    codes: list[int],
    field: str = "category",
) -> list[dict[str, Any]]:
    """Mean and SD of each metric's five-reference item score per code of `field`, pooled over models (None when
    there are too few items). `scores` maps (model, sample_id) to {metric: score}."""
    out = []
    for code in codes:
        items = [row for row in rows if row[field] == code]
        for metric in metrics:
            values = np.array([scores[(row["model_key"], row["sample_id"])][metric] for row in items])
            out.append(
                {
                    field: code,
                    "metric": metric,
                    "num_items": len(values),
                    "mean": float(values.mean()) if len(values) else None,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                }
            )
    return out


def split_disputed(text: str) -> list[str]:
    return [part.strip() for part in text.split(";") if part.strip()]


def disputed_summary(rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    out = []
    for model in models:
        items = [row for row in rows if row["model_key"] == model]
        phrases = [split_disputed(row["disputed_details"]) for row in items]
        out.append(
            {
                "model_key": model,
                "items_with_disputed_details": sum(bool(p) for p in phrases),
                "disputed_phrases": sum(len(p) for p in phrases),
                "total": len(items),
            }
        )
    return out


LIMITATIONS = [
    "One reviewer labelled every item, so inter-annotator agreement cannot be estimated and each label is a single "
    "judgement; more reviewers are left for future work.",
    "The reviewer had seen earlier captions from these models and knows their styles, so blinding is weaker than for "
    "a new reviewer; captions also carry style cues (E16).",
    "Categories describe agreement with the references, not correctness about the audio; disputed details are checked "
    "against the audio in E18.",
    "With 50 items per model, proportions are coarse (one item is 2 percentage points) and the Wilson intervals wide; "
    "the summary is descriptive and the four captions of a recording are not independent.",
]


def report_markdown(report: dict[str, Any]) -> str:
    """The annotation report: counts, shares and settings only, never caption text or notes."""
    names, categories = report["display_names"], report["categories"]
    run_info = report["run_info"]
    models = list(names)
    summary = {(r["model_key"], r["category"]): r for r in report["summary"]}
    lines = [
        "# Qualitative annotation (E17)",
        "",
        f"Generated by `scripts/analyze_annotations.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        "## Inputs",
        "",
        *[f"- `{item['path']}` (sha256 `{item['sha256']}`)" for item in report["inputs"].values()],
        f"- Blinding key SHA-256 matches `sample_manifest.json`: {report['key_verified']}.",
        f"- {report['num_items']} items ({report['num_recordings']} recordings x {len(models)} models), one reviewer, "
        "guidelines in `outputs/qualitative/annotation_guidelines.md`.",
        "",
        "## Agreement categories by model",
        "",
        f"Count and share of the model's {report['items_per_model']} captions, with Wilson "
        f"{report['confidence']:.0%} intervals.",
        "",
        "| Category | " + " | ".join(names[m] for m in models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for category in categories:
        code = category["code"]
        cells = []
        for model in models:
            row = summary[(model, code)]
            cells.append(f"{row['count']} ({row['share']:.0%}) [{row['ci_low']:.0%}, {row['ci_high']:.0%}]")
        lines.append(f"| {code}. {category['name']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Disputed details",
        "",
        "Captions with at least one phrase that no reference supports (`disputed_details`), and the number of phrases; "
        "these are the candidates for audio verification (E18).",
        "",
        "| Model | Captions with disputed details | Disputed phrases |",
        "|---|---|---|",
        *[
            f"| {names[r['model_key']]} | {r['items_with_disputed_details']} of {r['total']} | "
            f"{r['disputed_phrases']} |"
            for r in report["disputed"]
        ],
        "",
        "## Five-reference scores by category",
        "",
        "Mean item score (E11) of the captions in each category, pooled over models; descriptive only.",
        "",
        "| Category | Items | " + " | ".join(report["metric_labels"][m] for m in report["metrics"]) + " |",
        "|---|---|" + "---|" * len(report["metrics"]),
    ]
    by = {(r["category"], r["metric"]): r for r in report["scores_by_category"]}
    for category in categories:
        code = category["code"]
        count = by[(code, report["metrics"][0])]["num_items"]
        cells = [f"{by[(code, m)]['mean']:.3f}" if count else "n/a" for m in report["metrics"]]
        lines.append(f"| {code} | {count} | " + " | ".join(cells) + " |")
    lines += ["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""]
    lines += ["## Outputs", "", *[f"- `{path}`" for path in report["outputs"].values()], ""]
    return "\n".join(lines)
