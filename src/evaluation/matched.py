"""Matched four-reference comparison of model and human captions (E14): the human baseline from E13, model-human
agreement patterns and the report.

Each model caption is scored against the same `drop_k` subsets as the held-out human captions of E13, and both sides are
averaged over the five subsets in the same way, so model and human scores share references, metric implementations,
aggregation and bootstrap resamples. The human baseline is read from E13's committed tables, not rescored. The report
holds counts, IDs and scores only, never caption text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation import references as conditions
from src.evaluation.calibration import overlap_keep_masks, overlap_sample
from src.evaluation.scoring import ScoreSet, read_score_sets
from src.evaluation.summary import (
    ALL_RECORDINGS,
    POSITION_MEAN,
    VALID_FOR_ALL,
    _num,
    average_subsets,
    difference_rows,
    masked_average,
    matched_difference_rows,
    restrict,
    sensitivity_rows,
    summary_rows,
)
from src.metrics.registry import METRICS

HUMAN = "human"
VERDICTS = ("higher", "not_different", "lower")
MATCHED_SENSITIVITY_COLUMNS = (
    "sample", "condition", "subset", "metric", "order_all", "order_sample", "order_changed", "max_abs_mean_change",
    "significance_changes",
)  # fmt: skip

COMPARABILITY = [
    "Model and human captions are scored against the same four references (subset drop_k, the references left after "
    "removing CSV position k) and averaged over the five subsets in the same way; only these matched scores are "
    "compared. Four-reference scores are not comparable with the five-reference model scores of E12.",
    "Reference-free scores (CLAP similarity, MACE audio-only, fluency-error rates) do not depend on the references: a "
    "model's value is the same in every subset and equals its E12 value, while the human value averages five "
    "different captions.",
    "A model caption is never among its references, but a held-out human caption may have a near-paraphrase among "
    "its four (E13). The near-paraphrase samples keep, for model and human alike, only the (recording, subset) pairs "
    "whose held-out caption has closest-reference overlap below the threshold.",
]


def load_human_baseline(
    scores_path: Path, corpus_path: Path, report_path: Path, subsets: list[str], sample_ids: list[str]
) -> list[ScoreSet]:
    """E13's per-subset human score sets on `sample_ids`, checked against E13's report.

    The tables must hold exactly the report's subsets, metrics and recordings, and their `mean_drop_1_5` average must
    reproduce the report's baseline (corpus value and item mean) exactly. A recording subset (smoke test) is taken
    afterwards, as in `restrict`.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["condition"] != conditions.HUMAN_LOO or report.get("smoke_test_limit"):
        raise ValueError(f"{report_path} is not a full {conditions.HUMAN_LOO} run")
    sets = [s for s in read_score_sets(scores_path, corpus_path) if s.system == HUMAN]
    if [s.subset for s in sets] != subsets or report["subsets"] != subsets:
        raise ValueError(f"Human subsets {[s.subset for s in sets]} (report {report['subsets']}), expected {subsets}")
    if any(s.condition != conditions.HUMAN_LOO or list(s.scores) != report["metrics"] for s in sets):
        raise ValueError(f"{scores_path}: condition or metrics differ from {report_path}")
    averaged = average_subsets(sets)  # also checks that every subset scores the same recordings
    if len(averaged.sample_ids) != report["inputs"]["recordings"]:
        raise ValueError(f"{scores_path}: {len(averaged.sample_ids)} recordings, report has {report['inputs']}")
    expected = {
        r["metric"]: (r["corpus"], r["item_mean"])
        for r in report["summary"]
        if (r["subset"], r["sample"]) == (POSITION_MEAN, ALL_RECORDINGS)
    }
    for name, score in averaged.scores.items():
        if (_num(score.corpus), _num(float(score.items.mean()))) != expected[name]:
            raise ValueError(f"{scores_path}: {name} does not reproduce the baseline in {report_path}")
    if averaged.sample_ids == list(sample_ids):
        return sets
    return [restrict([s], sample_ids)[0] for s in sets]


def verdict(row: dict[str, Any]) -> str:
    """A model-minus-human difference row as higher, not_different or lower (95% CI, uncorrected)."""
    if not row["excludes_zero"]:
        return "not_different"
    return "higher" if float(row["difference"]) > 0 else "lower"


def verdict_rows(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per sample and metric: which models are significantly above, not different from, or below the human baseline,
    and whether every model falls on the same side (`uniform`)."""
    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    for row in matched:
        groups.setdefault((row["sample"], row["metric"]), {v: [] for v in VERDICTS})[verdict(row)].append(
            row["system_a"]
        )
    return [
        {
            "sample": sample,
            "metric": metric,
            **{v: ";".join(systems) for v, systems in by_verdict.items()},
            "uniform": sum(bool(systems) for systems in by_verdict.values()) == 1,
        }
        for (sample, metric), by_verdict in groups.items()
    ]


def matched_tables(
    score_sets: list[ScoreSet],
    human_sets: list[ScoreSet],
    valid_ids: list[str],
    overlaps: dict[str, list[float]],
    thresholds: list[float],
    boot: tuple[int, float, int],
) -> dict[str, Any]:
    """E14 report sections from the per-subset model score sets (models in input order) and E13's human sets.

    Samples: every recording; the recordings where every model's caption is valid; and, per overlap threshold, the
    (recording, subset) pairs whose held-out human caption is below it, for models and human alike. Per sample:
    model and human summaries, paired differences between models and model-minus-human differences, all on the same
    resamples. The sensitivity rows compare each other sample with every recording, human included.
    """
    subsets = [s.subset for s in human_sets]
    models = list(dict.fromkeys(s.system for s in score_sets))
    by_model = {model: [s for s in score_sets if s.system == model] for model in models}
    if any([s.subset for s in sets] != subsets for sets in by_model.values()):
        raise ValueError(f"Every model needs the human subsets {subsets}")
    means = [average_subsets(by_model[model]) for model in models]
    human = average_subsets(human_sets)
    samples = [
        (ALL_RECORDINGS, "every recording", means, human),
        (
            VALID_FOR_ALL,
            "recordings where every model's caption is valid (the malformed GPT captions excluded)",
            restrict(means, valid_ids),
            restrict([human], valid_ids)[0],
        ),
    ]
    for threshold in thresholds:
        keep = overlap_keep_masks(overlaps, human.sample_ids, subsets, threshold)
        samples.append(
            (
                overlap_sample(threshold),
                f"(recording, subset) pairs whose held-out human caption has closest-reference Jaccard < {threshold}",
                [masked_average(by_model[model], keep) for model in models],
                masked_average(human_sets, keep),
            )
        )
    summary = []
    for index in range(len(subsets)):
        summary += summary_rows([by_model[model][index] for model in models], ALL_RECORDINGS, *boot)
    differences, matched, tables = [], [], {}
    for name, _, model_sets, human_set in samples:
        rows = summary_rows(model_sets, name, *boot) + summary_rows([human_set], name, *boot)
        diffs = difference_rows(model_sets, name, *boot)
        vs_human = matched_difference_rows(model_sets, human_set, name, *boot)
        summary += rows
        differences += diffs
        matched += vs_human
        tables[name] = (rows, diffs + vs_human)
    full_summary, full_differences = tables[ALL_RECORDINGS]
    sensitivity = [
        {"sample": name, **row}
        for name, *_ in samples[1:]
        for row in sensitivity_rows(full_summary, tables[name][0], full_differences, tables[name][1])
    ]
    return {
        "summary": summary,
        "differences": differences,
        "matched": matched,
        "verdicts": verdict_rows(matched),
        "sensitivity": {
            "samples": [
                {"sample": name, "num_items": len(human_set.sample_ids), "description": description}
                for name, description, _, human_set in samples[1:]
            ],
            "rows": sensitivity,
        },
    }


def _fmt(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if value != "" else "n/a"


def _pattern(row: dict[str, Any], names: dict[str, str]) -> str:
    parts = [
        f"{label}: {', '.join(names.get(s, s) for s in row[v].split(';'))}"
        for v, label in zip(VERDICTS, ("above", "n.s.", "below"), strict=True)
        if row[v]
    ]
    return "; ".join(parts)


def _main_metrics(report: dict[str, Any]) -> list:
    return [m for m in METRICS if m.family != "auxiliary" and m.name in report["metrics"]]


def _scores_section(report: dict[str, Any], names: dict[str, str]) -> list[str]:
    systems = [*report["display_names"], HUMAN]
    rows = {
        (r["system"], r["metric"]): r
        for r in report["summary"]
        if (r["subset"], r["sample"]) == (POSITION_MEAN, ALL_RECORDINGS)
    }
    lines = [
        "## Scores (mean over the five subsets)",
        "",
        f"`{POSITION_MEAN}`: per recording, the mean of the five subset scores; the corpus value is the mean of the "
        "five per-subset corpus values. CI of the per-item mean in brackets; for BLEU and METEOR the item mean is in "
        "parentheses.",
        "",
        "| Metric | " + " | ".join(names[s] for s in systems) + " |",
        "|---|" + "---|" * len(systems),
    ]
    for metric in _main_metrics(report):
        cells = []
        for system in systems:
            row = rows[(system, metric.name)]
            mean = "" if metric.corpus_is_item_mean else f" ({_fmt(row['item_mean'])})"
            cells.append(f"{_fmt(row['corpus'])}{mean} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]")
        lines.append(f"| {metric.label} | " + " | ".join(cells) + " |")
    return lines


def _calibration_section(report: dict[str, Any], names: dict[str, str]) -> list[str]:
    models = list(report["display_names"])
    diffs = {(r["metric"], r["system_a"]): r for r in report["matched"] if r["sample"] == ALL_RECORDINGS}
    verdicts = {r["metric"]: r for r in report["verdicts"] if r["sample"] == ALL_RECORDINGS}
    main = _main_metrics(report)
    lines = [
        "",
        "## Model minus human (RQ2 calibration table)",
        "",
        f"Paired item-mean difference, model minus human, on `{POSITION_MEAN}`; `*` marks a 95% CI that excludes 0 "
        "(per comparison, uncorrected for multiple comparisons). The last column groups the models by whether they "
        "are significantly above (higher values), not different from, or below the human baseline.",
        "",
        "| Metric | " + " | ".join(f"{names[m]} - Human" for m in models) + " | Pattern |",
        "|---|" + "---|" * (len(models) + 1),
    ]
    for metric in main:
        cells = []
        for model in models:
            row = diffs[(metric.name, model)]
            cells.append(f"{float(row['difference']):+.4f}{'*' if row['excludes_zero'] else ''}")
        lines.append(f"| {metric.label} | " + " | ".join(cells) + f" | {_pattern(verdicts[metric.name], names)} |")
    split = [m.label for m in main if not verdicts[m.name]["uniform"]]
    reached = [m.label for m in main if verdicts[m.name]["higher"] or verdicts[m.name]["not_different"]]
    lines += [
        "",
        f"Metrics where the models fall on different sides of the human baseline: **{', '.join(split) or 'none'}**.",
        "",
        f"Metrics where at least one model is not significantly below the human baseline: "
        f"**{', '.join(reached) or 'none'}**.",
    ]
    return lines


def _model_differences_section(report: dict[str, Any], names: dict[str, str]) -> list[str]:
    rows = [r for r in report["differences"] if r["sample"] == ALL_RECORDINGS]
    pairs = list(dict.fromkeys((r["system_a"], r["system_b"]) for r in rows))
    diffs = {(r["metric"], r["system_a"], r["system_b"]): r for r in rows}
    lines = [
        "",
        "## Paired differences between models",
        "",
        f"Item-mean difference on `{POSITION_MEAN}` (row model minus column model, in model order); `*` as above.",
        "",
        "| Metric | " + " | ".join(f"{names[a]} - {names[b]}" for a, b in pairs) + " |",
        "|---|" + "---|" * len(pairs),
    ]
    for metric in _main_metrics(report):
        cells = []
        for a, b in pairs:
            row = diffs[(metric.name, a, b)]
            cells.append(f"{float(row['difference']):+.4f}{'*' if row['excludes_zero'] else ''}")
        lines.append(f"| {metric.label} | " + " | ".join(cells) + " |")
    return lines


def _subset_section(report: dict[str, Any], names: dict[str, str]) -> list[str]:
    models = list(report["display_names"])
    subsets = report["subsets"]
    corpus = {
        (r["system"], r["subset"], r["metric"]): float(r["corpus"])
        for r in report["summary"]
        if r["sample"] == ALL_RECORDINGS and r["subset"] in subsets
    }
    lines = [
        "",
        "## Spread over subsets",
        "",
        "Range (max - min) of each model's five per-subset corpus values.",
        "",
        "| Metric | " + " | ".join(names[m] for m in models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for metric in _main_metrics(report):
        cells = []
        for model in models:
            values = [corpus[(model, s, metric.name)] for s in subsets]
            cells.append(f"{max(values) - min(values):.4f}")
        lines.append(f"| {metric.label} | " + " | ".join(cells) + " |")
    return lines


def _sensitivity_section(report: dict[str, Any]) -> list[str]:
    samples = report["sensitivity"]["samples"]
    lines = [
        "",
        "## Sensitivity checks",
        "",
        "Recomputed from the per-item scores without rescoring, with new resamples of each sample; models and human "
        "are restricted in the same way. Item means (BLEU and METEOR have no corpus value on a subsample). Orders "
        "include the human baseline; significance changes cover model pairs and model-human pairs.",
        "",
        "| Sample | Recordings | Description |",
        "|---|---|---|",
        *[f"| `{s['sample']}` | {s['num_items']} | {s['description']} |" for s in samples],
    ]
    by_key = {(r["sample"], r["metric"]): r for r in report["sensitivity"]["rows"]}
    for sample in samples:
        name = sample["sample"]
        main = _main_metrics(report)
        changed = [
            m.label
            for m in main
            if by_key[(name, m.name)]["order_changed"] or by_key[(name, m.name)]["significance_changes"]
        ]
        lines += [
            "",
            f"### `{name}`",
            "",
            f"Metrics where the order or a significance changes: **{', '.join(changed) or 'none'}**.",
            "",
            "| Metric | Order (all) | Order (sample) | Largest mean change | Significance changes |",
            "|---|---|---|---|---|",
        ]
        for metric in main:
            row = by_key[(name, metric.name)]
            lines.append(
                f"| {metric.label} | {row['order_all']} | {row['order_sample']} | "
                f"{_fmt(row['max_abs_mean_change'], 5)} | {row['significance_changes'] or 'none'} |"
            )
    return lines


def report_markdown(report: dict[str, Any]) -> str:
    run_info, inputs, human = report["run_info"], report["inputs"], report["inputs"]["human_baseline"]
    names = {**report["display_names"], HUMAN: "Human"}
    lines = [
        "# Matched four-reference evaluation (E14)",
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
        f"- {inputs['recordings']} recordings x {len(report['display_names'])} models x {len(report['subsets'])} "
        f"subsets ({', '.join(report['subsets'])}), {inputs['references_per_subset']} references each; audio-aware "
        "metrics included. Every caption is scored as returned after the E10 normalization.",
        f"- Captions: `{inputs['captions']['path']}` (sha256 `{inputs['captions']['sha256']}`); references: "
        f"`{inputs['references']['path']}` (sha256 `{inputs['references']['sha256']}`).",
        f"- Human baseline: E13 tables `{human['scores']['path']}` (sha256 `{human['scores']['sha256']}`) and "
        f"`{human['corpus']['path']}`, from commit `{human['git_commit']}`; not rescored. Their average reproduces "
        f"`{human['report']}` exactly.",
        f"- Bootstrap: {report['bootstrap']['num_resamples']} paired resamples of recordings, "
        f"{report['bootstrap']['confidence']:.0%} percentile intervals, seed {report['bootstrap']['seed']} (the same "
        "resamples as E12 and E13 on the same recordings).",
        "",
        *_scores_section(report, names),
        *_calibration_section(report, names),
        *_model_differences_section(report, names),
        *_subset_section(report, names),
        *_sensitivity_section(report),
    ]
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
