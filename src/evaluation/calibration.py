"""Human leave-one-out calibration (E12): checks on the held-out splits, and the calibration report.

The caption at CSV position k of each recording is scored against the other four (subset `drop_k`), for k = 1..5, so
every human caption is scored once. The report holds counts, IDs and scores only, never caption text.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation import references as conditions
from src.evaluation.summary import ALL_RECORDINGS, POSITION_MEAN
from src.metrics.preprocess import check_texts
from src.metrics.registry import METRICS

COMPARABILITY = [
    "Human captions are scored against four references, so their reference-based scores are compared only with model "
    "captions scored against the same drop_k subsets (E13), never with the five-reference model scores (E11).",
    "Reference-free scores (CLAP similarity, MACE audio-only, fluency-error rates) do not depend on the references and "
    "can be compared directly with the E11 model scores.",
    "A CSV position is not a fixed annotator: Clotho's captions for one recording come from different annotators, in "
    "an arbitrary order. Differences between positions show the spread of the baseline, not annotator effects.",
]


def calibration_checks(references: dict[str, list[str]], num_positions: int) -> list[dict[str, Any]]:
    """Checks that must all pass before scoring; each is {name, passed, detail}."""
    checks = []

    def add(name: str, errors: list[str]) -> None:
        detail = f"{len(errors)} problems: {errors[:3]}" if errors else "ok"
        checks.append({"name": name, "passed": not errors, "detail": detail})

    add(
        f"every recording has {num_positions} references",
        [f"{sid}: {len(refs)}" for sid, refs in references.items() if len(refs) != num_positions],
    )
    texts = [error for sid, refs in references.items() for error in check_texts(refs, sid)]
    add("every reference is a non-empty single line", texts)
    if not checks[0]["passed"]:
        return checks
    subsets = conditions.subsets(conditions.HUMAN_LOO, num_positions)
    for subset in subsets:
        errors = conditions.check_split(references, subset)
        add(f"{subset}: held-out caption is not in its reference set (by position)", errors)
    held_out = [conditions.held_out_captions(references, subset) for subset in subsets]
    add(
        "the held-out captions of each recording are exactly its references, in CSV order",
        [sid for sid, refs in references.items() if [h[sid] for h in held_out] != refs],
    )
    return checks


def duplicate_held_out(tokenized: dict[str, list[str]]) -> list[str]:
    """`sample_id:position` of held-out captions identical, after tokenization, to one of their own references."""
    found = []
    for sid, refs in tokenized.items():
        for index, caption in enumerate(refs):
            if caption in refs[:index] + refs[index + 1 :]:
                found.append(f"{sid}:{index + 1}")
    return found


def closest_reference_overlap(tokenized: dict[str, list[str]]) -> dict[str, list[float]]:
    """Per recording and position, the word-set Jaccard between that caption and its closest other reference."""
    overlaps = {}
    for sid, refs in tokenized.items():
        words = [set(ref.split()) for ref in refs]
        overlaps[sid] = [
            max(len(words[i] & words[j]) / len(words[i] | words[j]) for j in range(len(words)) if j != i)
            for i in range(len(words))
        ]
    return overlaps


def overlap_keep_masks(
    overlaps: dict[str, list[float]], sample_ids: list[str], subsets: list[str], threshold: float
) -> dict[str, np.ndarray]:
    """`subset` -> per recording, True when the caption it holds out has closest-reference overlap below `threshold`."""
    return {
        subset: np.array([overlaps[sid][conditions.dropped_index(subset)] < threshold for sid in sample_ids])
        for subset in subsets
    }


def overlap_summary(overlaps: dict[str, list[float]], thresholds: list[float]) -> dict[str, Any]:
    values = np.array([value for per_recording in overlaps.values() for value in per_recording])
    return {
        "held_out_captions": int(values.size),
        "quartiles": [round(float(q), 4) for q in np.percentile(values, [25, 50, 75])],
        "thresholds": [
            {
                "threshold": threshold,
                "at_or_above": int((values >= threshold).sum()),
                "share": round(float((values >= threshold).mean()), 4),
                "recordings": sum(1 for v in overlaps.values() if max(v) >= threshold),
            }
            for threshold in thresholds
        ],
    }


def overlap_sample(threshold: float) -> str:
    """Summary `sample` name of the baseline restricted to held-out captions below `threshold`."""
    return f"closest_overlap_below_{threshold}"


def overlap_rows(overlaps: dict[str, list[float]]) -> list[dict[str, Any]]:
    return [
        {"sample_id": sid, "position": index + 1, "subset": f"drop_{index + 1}", "closest_reference_jaccard": repr(v)}
        for sid, values in overlaps.items()
        for index, v in enumerate(values)
    ]


OVERLAP_COLUMNS = ("sample_id", "position", "subset", "closest_reference_jaccard")


def _fmt(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if value != "" else "n/a"


def _overlap_section(report: dict[str, Any]) -> list[str]:
    overlap = report["overlap"]
    return [
        "## Near-paraphrase references",
        "",
        "Clotho's five captions of a recording often include near-paraphrase pairs, so a held-out caption may have a "
        "near-copy among its four references, which a model caption never has; this raises the human baseline. For "
        "each held-out caption: the word-set Jaccard (PTB tokens) with its closest other reference. Quartiles: "
        f"{', '.join(f'{q:.3f}' for q in overlap['quartiles'])} over {overlap['held_out_captions']} captions.",
        "",
        "| Closest-reference Jaccard | Held-out captions | Share | Recordings with one |",
        "|---|---|---|---|",
        *[
            f"| >= {t['threshold']} | {t['at_or_above']} | {t['share']:.1%} | {t['recordings']} |"
            for t in overlap["thresholds"]
        ],
        "",
        "Per-caption values: `" + report["outputs"]["overlap"] + "`.",
        "",
    ]


def _sensitivity_section(report: dict[str, Any]) -> list[str]:
    """Baseline item means on all held-out captions and on those below each overlap threshold."""
    samples = [(ALL_RECORDINGS, "All held-out captions")] + [
        (overlap_sample(t["threshold"]), f"Closest overlap < {t['threshold']}") for t in report["overlap"]["thresholds"]
    ]
    by_sample = {(r["sample"], r["metric"]): r for r in report["summary"] if r["subset"] == POSITION_MEAN}
    counts = {sample: by_sample[(sample, report["metrics"][0])]["num_items"] for sample, _ in samples}
    lines = [
        "",
        "## Sensitivity: held-out captions without a near-paraphrase reference",
        "",
        "Recomputed from the per-item scores without rescoring: each recording averages only its kept positions, and "
        "recordings with none kept are dropped (recordings in the header). Item means with bootstrap CIs, so BLEU and "
        "METEOR are item means here, not corpus values. The same masks apply to the model scores in E13.",
        "",
        "| Metric | " + " | ".join(f"{label} ({counts[sample]})" for sample, label in samples) + " |",
        "|---|" + "---|" * len(samples),
    ]
    for metric in (m for m in METRICS if m.name in report["metrics"] and m.family != "auxiliary"):
        cells = []
        for sample, _ in samples:
            row = by_sample[(sample, metric.name)]
            cells.append(f"{_fmt(row['item_mean'])} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]")
        lines.append(f"| {metric.label} | " + " | ".join(cells) + " |")
    return lines


def report_markdown(report: dict[str, Any]) -> str:
    run_info, inputs = report["run_info"], report["inputs"]
    subsets = report["subsets"]
    duplicates = str(len(report["duplicates"]))
    if report["duplicates"]:
        duplicates += f" ({', '.join(report['duplicates'][:10])})"
    rows = {(r["subset"], r["metric"]): r for r in report["summary"] if r["sample"] == ALL_RECORDINGS}
    lines = [
        "# Human leave-one-out calibration (E12)",
        "",
        f"Generated by `scripts/human_leave_one_out.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        *(
            [f"**Smoke test on the first {report['smoke_test_limit']} recordings only; not results.**", ""]
            if report.get("smoke_test_limit")
            else []
        ),
        "## Inputs",
        "",
        f"- {inputs['recordings']} recordings x {len(subsets)} held-out positions = {inputs['held_out_captions']} "
        f"human captions, each scored against the other {inputs['references_per_subset']}; audio-aware metrics "
        "included.",
        f"- References: `{inputs['references']['path']}` (sha256 `{inputs['references']['sha256']}`).",
        f"- Bootstrap: {report['bootstrap']['num_resamples']} resamples of recordings, "
        f"{report['bootstrap']['confidence']:.0%} percentile intervals, seed {report['bootstrap']['seed']} "
        "(the same resamples as E11 and E13, so differences with model scores stay paired).",
        "",
        "## Calibration checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
        *[f"| {c['name']} | {c['passed']} | {c['detail']} |" for c in report["checks"]],
        "",
        f"Held-out captions identical to one of their own references after PTB tokenization: {duplicates}. "
        "They are kept; a duplicate counts as a valid reference.",
        "",
        *_overlap_section(report),
        "## Human baseline",
        "",
        f"`{POSITION_MEAN}`: per recording, the mean of the five position scores; the corpus value is the mean of the "
        "five per-position corpus values. CI of the per-item mean in brackets; for BLEU and METEOR the item mean is in "
        "parentheses. The last columns are the per-position corpus values.",
        "",
        "| Metric | Family | Baseline [95% CI] | " + " | ".join(subsets) + " | Range |",
        "|---|---|---|" + "---|" * len(subsets) + "---|",
    ]
    for metric in (m for m in METRICS if m.name in report["metrics"]):
        row = rows[(POSITION_MEAN, metric.name)]
        mean = "" if metric.corpus_is_item_mean else f" ({_fmt(row['item_mean'])})"
        per_position = [float(rows[(s, metric.name)]["corpus"]) for s in subsets]
        lines.append(
            f"| {metric.label} | {metric.family} | {_fmt(row['corpus'])}{mean} [{_fmt(row['ci_low'])}, "
            f"{_fmt(row['ci_high'])}] | " + " | ".join(f"{v:.4f}" for v in per_position) + f" | "
            f"{max(per_position) - min(per_position):.4f} |"
        )
    lines += _sensitivity_section(report)
    timing = report["timing"]
    lines += [
        "",
        "## Run",
        "",
        f"- Scoring: {timing['scoring_seconds'] / 60:.1f} min on CPU; "
        f"model loading {timing['load_models_seconds']:.0f} s. "
        f"CLAP audio embeddings from the cache: {timing['clap_cache_hits']} of {inputs['recordings']}.",
        "",
        "| Step | Seconds | Calls |",
        "|---|---|---|",
        *[f"| {name} | {seconds:.1f} | {timing['calls'][name]} |" for name, seconds in timing["seconds"].items()],
        "",
        "## Interpretation and limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    return "\n".join(lines)
