"""Blinded qualitative review sample (E16).

A simple random sample of evaluation recordings (seeded, independent of input order and of model scores).
Presentation is randomized in separate streams derived from the same seed: the order of the recordings
(which defines the codes R01, R02, ...) and, per recording, the order of the models behind the labels A, B, C, ...
Review rows hold only the recording code, the references and the caption as scored; the key that maps items back to
recordings and models is written separately.
"""

from __future__ import annotations

import random
import string
from typing import Any

import numpy as np

from src.data.manifest import select_sample as _seeded_sample

REFERENCE_COLUMNS = tuple(f"reference_{k}" for k in range(1, 6))
REVIEW_COLUMNS = ("item_id", "recording_code", "caption_label", *REFERENCE_COLUMNS, "caption")
ANNOTATION_FIELDS = (
    "category", "disputed_details", "notes_sources", "notes_attributes", "notes_temporal", "notes_other",
)  # fmt: skip
ANNOTATION_COLUMNS = (*REVIEW_COLUMNS, *ANNOTATION_FIELDS)
KEY_COLUMNS = ("item_id", "recording_code", "caption_label", "sample_id", "model_key", "normalization")
CATEGORY_COLUMNS = ("code", "name", "description")


def select_sample(sample_ids: list[str], num_recordings: int, seed: int) -> list[str]:
    """Seeded simple random sample, sorted; the same IDs whatever the input order."""
    return _seeded_sample(sample_ids, num_recordings, seed)


def _stream(seed: int, name: str) -> random.Random:
    return random.Random(f"{seed}:{name}")


def blind(sample: list[str], models: list[str], seed: int) -> list[dict[str, str]]:
    """Key rows (one per recording and model) in presentation order: shuffled recordings coded R01..., and per
    recording a shuffled model order behind the labels A, B, ..."""
    if len(models) > len(string.ascii_uppercase):
        raise ValueError("Too many models for single-letter labels")
    order = sorted(sample)
    _stream(seed, "recording_order").shuffle(order)
    width = max(2, len(str(len(order))))
    models_rng = _stream(seed, "model_order")
    rows = []
    for index, sample_id in enumerate(order, start=1):
        code = f"R{index:0{width}d}"
        shuffled = list(models)
        models_rng.shuffle(shuffled)
        for label, model in zip(string.ascii_uppercase, shuffled, strict=False):
            rows.append(
                {"item_id": f"{code}{label}", "recording_code": code, "caption_label": label, "sample_id": sample_id}
                | {"model_key": model}
            )
    return rows


def review_rows(
    key: list[dict[str, str]], references: dict[str, list[str]], captions: dict[str, dict[str, dict[str, Any]]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Blinded review rows and the full key rows. `captions[model][sample_id]` is the consolidated caption record."""
    review, full_key = [], []
    for row in key:
        refs = references[row["sample_id"]]
        if len(refs) != len(REFERENCE_COLUMNS):
            raise ValueError(f"{row['sample_id']}: {len(refs)} references, expected {len(REFERENCE_COLUMNS)}")
        record = captions[row["model_key"]][row["sample_id"]]
        review.append(
            {
                "item_id": row["item_id"],
                "recording_code": row["recording_code"],
                "caption_label": row["caption_label"],
                **dict(zip(REFERENCE_COLUMNS, refs, strict=True)),
                "caption": record["caption"],
            }
        )
        full_key.append({**row, "normalization": record["normalization"]})
    return review, full_key


def annotation_template(review: list[dict[str, str]]) -> list[dict[str, str]]:
    blank = dict.fromkeys(ANNOTATION_FIELDS, "")
    return [{**row, **blank} for row in review]


def has_annotations(rows: list[dict[str, str]]) -> bool:
    """Whether any annotation field of a template is filled in (such a template is never overwritten)."""
    return any(row.get(field, "").strip() for row in rows for field in ANNOTATION_FIELDS)


def review_sheet_markdown(review: list[dict[str, str]]) -> str:
    """A reading view of the review rows: per recording code, its references and the labelled captions. Labels are
    entered in the annotation template, not here."""
    lines = [
        "# Qualitative review sheet (E16-E17)",
        "",
        "Reading view of `review_items.csv`. Enter labels in the annotation template (`annotations_reviewer_1.csv`), "
        "following `annotation_guidelines.md`.",
        "",
    ]
    by_code: dict[str, list[dict[str, str]]] = {}
    for row in review:
        by_code.setdefault(row["recording_code"], []).append(row)
    for code, rows in by_code.items():
        lines += [f"## {code}", "", "References:", ""]
        lines += [f"{k}. {rows[0][column]}" for k, column in enumerate(REFERENCE_COLUMNS, start=1)]
        lines += ["", "Captions:", ""]
        lines += [f"- **{row['item_id']}**: {row['caption']}" for row in rows]
        lines.append("")
    return "\n".join(lines)


def blinding_problems(review: list[dict[str, str]], model_terms: list[str], file_names: list[str]) -> list[str]:
    """Review cells that contain a model key or display name, or a recording's file name (case-insensitive)."""
    terms = [term.lower() for term in [*model_terms, *file_names] if term]
    return [
        f"{row['item_id']}: {column} contains {term!r}"
        for row in review
        for column, value in row.items()
        for term in terms
        if term in value.lower()
    ]


def sample_comparison(
    sample: list[str], population: list[str], values: dict[str, dict[str, np.ndarray]]
) -> list[dict[str, Any]]:
    """Mean of each per-recording quantity on the sample and on the population, and their standardized difference
    (in population SDs); descriptive only. `values[group][quantity]` is aligned with `population`."""
    index = {sid: i for i, sid in enumerate(population)}
    rows = [index[sid] for sid in sample]
    out = []
    for group, quantities in values.items():
        for quantity, array in quantities.items():
            array = np.asarray(array, dtype=np.float64)
            sd = float(array.std(ddof=1))
            difference = float(array[rows].mean() - array.mean())
            out.append(
                {
                    "group": group,
                    "quantity": quantity,
                    "sample_mean": float(array[rows].mean()),
                    "population_mean": float(array.mean()),
                    "population_sd": sd,
                    "standardized_difference": difference / sd if sd > 0 else float("nan"),
                }
            )
    return out


def report_markdown(report: dict[str, Any]) -> str:
    """Sample report: settings, IDs, counts and aggregates only, never caption text or the blinding key."""
    run_info, manifest = report["run_info"], report["manifest"]
    lines = [
        "# Qualitative review sample (E16)",
        "",
        f"Generated by `scripts/prepare_qualitative.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        "## Sample",
        "",
        f"- {manifest['num_recordings']} recordings drawn at random (seed {manifest['seed']}) from the "
        f"{manifest['population']} evaluation recordings, independently of model scores; "
        f"{manifest['num_items']} model captions ({len(report['models'])} per recording).",
        "- Each recording has its own random model order behind the labels A-D, and recordings are presented in "
        "random order (codes R01...). Review sheets contain no file names, model names or scores.",
        f"- Blinding key: `{manifest['key']['path']}` (not committed until the E17 labels are final), "
        f"sha256 `{manifest['key']['sha256']}`.",
        f"- Captions as scored, by normalization: {report['normalization']}.",
        f"- Annotation templates: {', '.join(f'`{p}`' for p in manifest['templates'])}.",
        "",
        "Sample IDs (sorted): " + ", ".join(f"`{sid}`" for sid in manifest["sample_ids"]),
        "",
        "## Representativeness (descriptive)",
        "",
        "Means on the sample and on all recordings; the standardized difference is (sample mean - population mean) / "
        "population SD of the per-recording values. Model scores are five-reference item means (E11).",
        "",
        "| Group | Quantity | Sample | All | Std. difference |",
        "|---|---|---|---|---|",
        *[
            f"| {r['group']} | {r['quantity']} | {r['sample_mean']:.3f} | {r['population_mean']:.3f} | "
            f"{r['standardized_difference']:+.2f} |"
            for r in report["comparison"]
        ],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
        "## Outputs",
        "",
        *[f"- `{path}`" for path in report["outputs"].values()],
        "",
    ]
    return "\n".join(lines)


LIMITATIONS = [
    "Blinding is imperfect: style can reveal a model (GPT's JSON-like captions are shown as scored, many Qwen captions "
    "begin with 'The audio', and the models differ in typical length; see E05-E09 and E14).",
    "One reviewer annotates the sample, so inter-annotator agreement cannot be estimated and every label reflects a "
    "single judgement; more reviewers are left for future work.",
    "50 recordings give coarse category proportions per model (one caption is 2 percentage points); the review is "
    "exploratory and not used to estimate dataset-wide rates.",
]
