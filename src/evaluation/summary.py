"""Model-level summaries of one reference condition (E12, reused in E13-E14): bootstrap CIs, paired model differences
(and differences from a baseline scored on the same references) and the sensitivity check on a subset of recordings.

Everything works on per-item scores that are already computed, so a subset of recordings needs no rescoring. On a
subset, metrics whose corpus value comes from corpus statistics (BLEU, METEOR) have no corpus value, and CIDEr-D keeps
the document frequencies of the full call's references. The sensitivity check compares item means, the statistic the
bootstrap is on, so the two samples are compared like with like.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np

from src.evaluation.aggregate import bootstrap_difference, bootstrap_indices, bootstrap_mean
from src.evaluation.scoring import ScoreSet
from src.metrics.registry import BY_NAME, METRICS, Score

ALL_RECORDINGS = "all"
VALID_FOR_ALL = "valid_for_all_models"
POSITION_MEAN = "mean_drop_1_5"  # the five drop_k subsets averaged (E13-E14)

SUMMARY_COLUMNS = (
    "system", "condition", "subset", "sample", "num_items", "metric", "corpus", "item_mean", "ci_low", "ci_high"
)  # fmt: skip
DIFFERENCE_COLUMNS = (
    "condition", "subset", "sample", "num_items", "metric", "system_a", "system_b", "difference", "ci_low", "ci_high",
    "excludes_zero",
)  # fmt: skip
SENSITIVITY_COLUMNS = (
    "condition", "subset", "metric", "order_all", "order_sample", "order_changed", "max_abs_mean_change",
    "significance_changes",
)  # fmt: skip


def _num(value: float) -> str:
    return "" if value is None or math.isnan(value) else repr(float(value))


def _check_aligned(score_sets: list[ScoreSet]) -> list[str]:
    if not score_sets:
        raise ValueError("No score sets")
    first = score_sets[0]
    for score_set in score_sets[1:]:
        key = (score_set.condition, score_set.subset, score_set.sample_ids)
        if key != (first.condition, first.subset, first.sample_ids):
            raise ValueError(f"{score_set.system} is not scored on the same condition and recordings as {first.system}")
        if list(score_set.scores) != list(first.scores):
            raise ValueError(f"{score_set.system} has different metrics from {first.system}")
    return first.sample_ids


def average_subsets(score_sets: list[ScoreSet], subset: str = POSITION_MEAN) -> ScoreSet:
    """One system's per-subset score sets averaged (proposal §5.2): per recording, the mean of its subset scores; the
    corpus value, the mean of the per-subset corpus values (so corpus BLEU and METEOR stay corpus statistics)."""
    if not score_sets:
        raise ValueError("No score sets")
    first = score_sets[0]
    for score_set in score_sets[1:]:
        if (score_set.system, score_set.condition, score_set.sample_ids) != (
            first.system,
            first.condition,
            first.sample_ids,
        ):
            raise ValueError(f"{score_set.subset} is not the same system, condition and recordings as {first.subset}")
        if list(score_set.scores) != list(first.scores):
            raise ValueError(f"{score_set.subset} has different metrics from {first.subset}")
    if len({score_set.subset for score_set in score_sets}) != len(score_sets):
        raise ValueError("Duplicate subsets")
    scores = {
        name: Score(
            float(np.mean([score_set.scores[name].corpus for score_set in score_sets])),
            np.mean([score_set.scores[name].items for score_set in score_sets], axis=0),
        )
        for name in first.scores
    }
    return ScoreSet(first.system, first.condition, subset, list(first.sample_ids), scores)


def masked_average(score_sets: list[ScoreSet], keep: dict[str, np.ndarray], subset: str = POSITION_MEAN) -> ScoreSet:
    """`average_subsets` over the kept (recording, subset) scores only (E13-E14 near-paraphrase sensitivity).

    `keep[subset]` is a boolean array over the recordings. Each recording's score is the mean of its kept subsets;
    recordings with none kept are dropped. Corpus-statistic metrics (BLEU, METEOR) get no corpus value, as in
    `restrict`.
    """
    averaged = average_subsets(score_sets, subset)  # checks alignment
    mask = np.stack([np.asarray(keep[score_set.subset], dtype=bool) for score_set in score_sets])
    if mask.shape != (len(score_sets), len(averaged.sample_ids)):
        expected = (len(score_sets), len(averaged.sample_ids))
        raise ValueError(f"Mask shape {mask.shape} does not match {expected} (subsets x recordings)")
    counts = mask.sum(axis=0)
    rows = counts > 0
    scores = {}
    for name in averaged.scores:
        values = np.stack([score_set.scores[name].items for score_set in score_sets])
        items = (np.where(mask, values, 0.0).sum(axis=0)[rows]) / counts[rows]
        corpus = float(items.mean()) if BY_NAME[name].corpus_is_item_mean and items.size else math.nan
        scores[name] = Score(corpus, items)
    ids = [sid for sid, kept in zip(averaged.sample_ids, rows, strict=True) if kept]
    return ScoreSet(averaged.system, averaged.condition, subset, ids, scores)


def restrict(score_sets: list[ScoreSet], sample_ids: list[str]) -> list[ScoreSet]:
    """The same score sets on `sample_ids` only, kept in the score sets' recording order."""
    order = _check_aligned(score_sets)
    wanted = set(sample_ids)
    if missing := sorted(wanted - set(order)):
        raise ValueError(f"{len(missing)} recordings are not scored, e.g. {missing[:3]}")
    keep = np.array([sid in wanted for sid in order])
    ids = [sid for sid in order if sid in wanted]
    restricted = []
    for score_set in score_sets:
        scores = {}
        for name, score in score_set.scores.items():
            items = score.items[keep]
            corpus = float(items.mean()) if BY_NAME[name].corpus_is_item_mean else math.nan
            scores[name] = Score(corpus, items)
        restricted.append(ScoreSet(score_set.system, score_set.condition, score_set.subset, ids, scores))
    return restricted


def summary_rows(
    score_sets: list[ScoreSet], sample: str, num_resamples: int, confidence: float, seed: int
) -> list[dict[str, Any]]:
    """Per system and metric: corpus value, item mean and its percentile CI; every system uses the same resamples."""
    indices = bootstrap_indices(len(_check_aligned(score_sets)), num_resamples, seed)
    rows = []
    for score_set in score_sets:
        for name, score in score_set.scores.items():
            ci = bootstrap_mean(score.items, indices, confidence)
            rows.append(
                {
                    "system": score_set.system,
                    "condition": score_set.condition,
                    "subset": score_set.subset,
                    "sample": sample,
                    "num_items": len(score.items),
                    "metric": name,
                    "corpus": _num(score.corpus),
                    "item_mean": _num(ci["mean"]),
                    "ci_low": _num(ci["ci_low"]),
                    "ci_high": _num(ci["ci_high"]),
                }
            )
    return rows


def difference_rows(
    score_sets: list[ScoreSet], sample: str, num_resamples: int, confidence: float, seed: int
) -> list[dict[str, Any]]:
    """Paired item-mean differences for every pair of systems (in input order), with the summary's resamples."""
    indices = bootstrap_indices(len(_check_aligned(score_sets)), num_resamples, seed)
    rows = []
    for name in score_sets[0].scores:
        for a, b in combinations(score_sets, 2):
            diff = bootstrap_difference(a.scores[name].items, b.scores[name].items, indices, confidence)
            rows.append(
                {
                    "condition": a.condition,
                    "subset": a.subset,
                    "sample": sample,
                    "num_items": len(a.sample_ids),
                    "metric": name,
                    "system_a": a.system,
                    "system_b": b.system,
                    "difference": _num(diff["mean"]),
                    "ci_low": _num(diff["ci_low"]),
                    "ci_high": _num(diff["ci_high"]),
                    "excludes_zero": diff["ci_low"] > 0 or diff["ci_high"] < 0,
                }
            )
    return rows


def matched_difference_rows(
    score_sets: list[ScoreSet], baseline: ScoreSet, sample: str, num_resamples: int, confidence: float, seed: int
) -> list[dict[str, Any]]:
    """Paired item-mean differences of every system minus `baseline`, with the summary's resamples (E14).

    The baseline may come from another condition (the human leave-one-out captions) but must be on the same subset,
    recordings and metrics, so both sides are scored against the same references.
    """
    ids = _check_aligned(score_sets)
    first = score_sets[0]
    if (baseline.subset, baseline.sample_ids) != (first.subset, ids):
        raise ValueError(f"{baseline.system} is not on the same subset and recordings as {first.system}")
    if list(baseline.scores) != list(first.scores):
        raise ValueError(f"{baseline.system} has different metrics from {first.system}")
    indices = bootstrap_indices(len(ids), num_resamples, seed)
    rows = []
    for name in first.scores:
        for score_set in score_sets:
            diff = bootstrap_difference(score_set.scores[name].items, baseline.scores[name].items, indices, confidence)
            rows.append(
                {
                    "condition": score_set.condition,
                    "subset": score_set.subset,
                    "sample": sample,
                    "num_items": len(ids),
                    "metric": name,
                    "system_a": score_set.system,
                    "system_b": baseline.system,
                    "difference": _num(diff["mean"]),
                    "ci_low": _num(diff["ci_low"]),
                    "ci_high": _num(diff["ci_high"]),
                    "excludes_zero": diff["ci_low"] > 0 or diff["ci_high"] < 0,
                }
            )
    return rows


def _order(rows: list[dict[str, Any]], metric: str) -> list[str]:
    """Systems by descending item mean (ties keep input order). Lower is better for the fluency-error rates."""
    values = [(row["system"], float(row["item_mean"])) for row in rows if row["metric"] == metric]
    return [system for system, _ in sorted(values, key=lambda item: -item[1])]


def sensitivity_rows(
    full_summary: list[dict[str, Any]],
    sample_summary: list[dict[str, Any]],
    full_differences: list[dict[str, Any]],
    sample_differences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per metric: whether the order of systems changes on the sample, and which pairwise CIs flip on excluding 0."""
    metrics = list(dict.fromkeys(row["metric"] for row in full_summary))
    full_means = {(r["system"], r["metric"]): float(r["item_mean"]) for r in full_summary}
    sample_means = {(r["system"], r["metric"]): float(r["item_mean"]) for r in sample_summary}
    sample_excludes = {(r["metric"], r["system_a"], r["system_b"]): r["excludes_zero"] for r in sample_differences}
    first = full_summary[0]
    rows = []
    for metric in metrics:
        full_order, sample_order = _order(full_summary, metric), _order(sample_summary, metric)
        changes = []
        for row in (r for r in full_differences if r["metric"] == metric):
            after = sample_excludes[(metric, row["system_a"], row["system_b"])]
            if after != row["excludes_zero"]:
                changes.append(f"{row['system_a']}-{row['system_b']}: {_yes(row['excludes_zero'])}->{_yes(after)}")
        shift = max(abs(full_means[key] - sample_means[key]) for key in full_means if key[1] == metric)
        rows.append(
            {
                "condition": first["condition"],
                "subset": first["subset"],
                "metric": metric,
                "order_all": " > ".join(full_order),
                "order_sample": " > ".join(sample_order),
                "order_changed": full_order != sample_order,
                "max_abs_mean_change": _num(shift),
                "significance_changes": "; ".join(changes),
            }
        )
    return rows


def _yes(value: bool) -> str:
    return "excludes 0" if value else "includes 0"


def _fmt(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if value != "" else "n/a"


def report_markdown(report: dict[str, Any]) -> str:
    """The evaluation report: numbers, IDs and settings only, never caption text."""
    names = report["display_names"]
    systems = list(names)
    main = [m for m in METRICS if m.family != "auxiliary" and m.name in report["metrics"]]
    full = {(r["system"], r["metric"]): r for r in report["summary"] if r["sample"] == ALL_RECORDINGS}
    run_info, inputs, sensitivity = report["run_info"], report["inputs"], report["sensitivity"]
    lines = [
        f"# {report['condition']} evaluation ({report['task']})",
        "",
        f"Generated by `scripts/evaluate_captions.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        *(
            [f"**Smoke test on the first {report['smoke_test_limit']} recordings only; not results.**", ""]
            if report.get("smoke_test_limit")
            else []
        ),
        "## Inputs",
        "",
        f"- {inputs['recordings']} recordings x {len(systems)} models; subset `{report['subset']}` "
        f"({inputs['references_per_recording']} references per recording); audio-aware metrics included.",
        f"- Captions: `{inputs['captions']['path']}` (sha256 `{inputs['captions']['sha256']}`); references: "
        f"`{inputs['references']['path']}` (sha256 `{inputs['references']['sha256']}`).",
        "- Every caption is scored as returned after the E10 normalization; no recording is excluded.",
        "",
        "| Model | Key | Normalization |",
        "|---|---|---|",
        *[f"| {names[s]} | `{s}` | {inputs['normalization'].get(s, {})} |" for s in systems],
        "",
        f"Bootstrap: {report['bootstrap']['num_resamples']} paired resamples of recordings, "
        f"{report['bootstrap']['confidence']:.0%} percentile intervals, seed {report['bootstrap']['seed']}.",
        "",
        "## Scores by model",
        "",
        "Corpus value, and the bootstrap CI of the per-item mean in brackets. For BLEU and METEOR the corpus value "
        "comes from corpus statistics and the item mean (in parentheses) is what the CI is on.",
        "",
        "| Metric | " + " | ".join(names[s] for s in systems) + " |",
        "|---|" + "---|" * len(systems),
    ]
    for metric in main:
        cells = []
        for system in systems:
            row = full[(system, metric.name)]
            mean = "" if metric.corpus_is_item_mean else f" ({_fmt(row['item_mean'])})"
            cells.append(f"{_fmt(row['corpus'])}{mean} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]")
        lines.append(f"| {metric.label} | " + " | ".join(cells) + " |")
    pairs = [(a, b) for a, b in combinations(systems, 2)]
    diffs = {
        (r["metric"], r["system_a"], r["system_b"]): r for r in report["differences"] if r["sample"] == ALL_RECORDINGS
    }
    lines += [
        "",
        "## Paired differences between models",
        "",
        "Item-mean difference (row model minus column model, in model order); `*` marks a 95% CI that excludes 0 "
        "(per comparison, uncorrected for multiple comparisons).",
        "",
        "| Metric | " + " | ".join(f"{names[a]} - {names[b]}" for a, b in pairs) + " |",
        "|---|" + "---|" * len(pairs),
    ]
    for metric in main:
        cells = []
        for a, b in pairs:
            row = diffs[(metric.name, a, b)]
            cells.append(f"{float(row['difference']):+.4f}{'*' if row['excludes_zero'] else ''}")
        lines.append(f"| {metric.label} | " + " | ".join(cells) + " |")
    by_metric = {row["metric"]: row for row in sensitivity["rows"]}
    changed = [m.label for m in main if by_metric[m.name]["order_changed"] or by_metric[m.name]["significance_changes"]]
    lines += [
        "",
        f"## Sensitivity check: {sensitivity['num_items']} recordings where every model's caption is valid",
        "",
        f"Item means recomputed from the per-item scores on the {sensitivity['num_items']} recordings without a "
        f"malformed caption ({sensitivity['excluded']} excluded), with new resamples of that sample. "
        f"Metrics where the model order or a pairwise significance changes: **{', '.join(changed) or 'none'}**.",
        "",
        "| Metric | Order (all) | Order (valid) | Order changed | Largest mean change | Significance changes |",
        "|---|---|---|---|---|---|",
    ]
    for metric in main:
        row = by_metric[metric.name]
        lines.append(
            f"| {metric.label} | {row['order_all']} | {row['order_sample']} | {row['order_changed']} | "
            f"{_fmt(row['max_abs_mean_change'], 5)} | {row['significance_changes'] or 'none'} |"
        )
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
