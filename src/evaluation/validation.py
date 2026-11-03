"""Metric validation report (E11): reproducibility, truncation, timing and provenance of the metric pipeline.

The report holds counts, scores, versions and checksums only, never caption text.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from typing import Any

import numpy as np

from src.evaluation.scoring import ScoreSet
from src.metrics.registry import METRICS

PACKAGES = (
    "aac-metrics",
    "msclap",
    "torch",
    "torchaudio",
    "transformers",
    "sentence-transformers",
    "torchmetrics",
    "huggingface-hub",
    "numpy",
    "pandas",
    "librosa",
    "soundfile",
)

LIMITATIONS = [
    "CIDEr-D, SPICE, SPIDEr and SPIDEr-FL depend on the reference set (CIDEr-D's document frequencies come from the "
    "references of each call), so five-reference (E12) and four-reference (E13/E14) scores are not directly "
    "comparable.",
    "BLEU and METEOR corpus values are computed from corpus statistics, not as the mean of per-item scores; "
    "bootstrap intervals are on the per-item mean and are reported next to the corpus value.",
    "BERTScore follows aac-metrics: roberta-large's last layer (24), no IDF, the maximum over references. Raw F1 is "
    "compressed near 1; the rescaled F1 (layer-24 baseline) is reported as a secondary column.",
    "FENSE, SPIDEr-FL and MACE multiply a caption's score by a fixed factor when the fluency-error detector flags it "
    "(probability above 0.9, or 0.97 for MACE). The detector also flags some short, plain captions, and its input is "
    "cut at 64 subwords.",
    "Malformed (JSON-like) captions are scored as returned after PTB tokenization: braces and quotes are removed but "
    "key names remain, which lowers lexical and semantic scores.",
    "MS-CLAP 2023 sees 7 s of audio at a time; clips are embedded whole as a duration-weighted mean of 7 s chunks, "
    "following the MACE paper. aac-metrics' own MACE instead uses one random 7 s crop, is text-only by default and "
    "returns the corpus mean as every per-item score.",
    "SPICE runs without its parse cache (the cache's native library is x86-only); this changes speed, not scores.",
    "Every metric runs on CPU in float32/float64; scores are reproducible on the same platform and versions but may "
    "differ in the last digits on other hardware.",
]


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def compare_runs(first: list[ScoreSet], second: list[ScoreSet]) -> dict[str, Any]:
    """Whether two runs produced bit-identical per-item and corpus scores."""
    if [(s.system, s.condition, s.subset) for s in first] != [(s.system, s.condition, s.subset) for s in second]:
        return {"identical": False, "values_compared": 0, "max_abs_difference": None, "differing": ["call order"]}
    compared, max_diff, differing = 0, 0.0, []
    for a, b in zip(first, second, strict=True):
        if a.sample_ids != b.sample_ids or list(a.scores) != list(b.scores):
            differing.append(f"{a.system}/{a.condition}/{a.subset}: different items or metrics")
            continue
        for name, score in a.scores.items():
            other = b.scores[name]
            compared += score.items.size + 1
            diff = max(float(np.max(np.abs(score.items - other.items))), abs(score.corpus - other.corpus))
            max_diff = max(max_diff, diff)
            if not (np.array_equal(score.items, other.items) and score.corpus == other.corpus):
                differing.append(f"{a.system}/{a.condition}/{a.subset}/{name}")
    return {
        "identical": not differing,
        "values_compared": compared,
        "max_abs_difference": max_diff,
        "differing": differing[:20],
    }


def extrapolate(seconds: dict[str, float], pilot_items: int, full_items: int) -> dict[str, Any]:
    """Scale per-step compute time linearly from the pilot to the full evaluation set.

    An upper bound: fixed per-call costs (JVM start-up, CoreNLP model loading in SPICE) are scaled up too.
    """
    factor = full_items / pilot_items
    steps = {name: round(value * factor, 1) for name, value in seconds.items() if name != "load_models"}
    return {
        "pilot_items": pilot_items,
        "full_items": full_items,
        "factor": round(factor, 3),
        "estimated_seconds": steps,
        "estimated_total_hours": round(sum(steps.values()) / 3600, 2),
    }


# Steps whose cost per call grows with the number of items in that call. The fluency-error detector and CLAP embed
# each distinct text or recording once per run, so they scale linearly with the data instead.
PER_CALL_STEPS = ("ptb_tokenize", "bleu", "meteor", "cider_d", "spice", "sbert", "bertscore")


def two_point_estimate(
    small: dict[str, float], small_items: int, large: dict[str, float], large_items: int, full_items: int, calls: int
) -> dict[str, Any]:
    """Estimate per-call steps from one call's time at two sizes: fixed cost + cost per item, times `calls` calls.

    `small` and `large` are seconds per call at `small_items` and `large_items` items.
    """
    steps = {}
    for name in PER_CALL_STEPS:
        if name in small and name in large:
            per_item = max((large[name] - small[name]) / (large_items - small_items), 0.0)
            fixed = max(small[name] - per_item * small_items, 0.0)
            steps[name] = {
                "fixed_seconds": round(fixed, 2),
                "seconds_per_item": round(per_item, 4),
                "estimated_seconds": round(calls * (fixed + per_item * full_items), 1),
            }
    total = sum(step["estimated_seconds"] for step in steps.values())
    return {"calls": calls, "steps": steps, "estimated_hours": round(total / 3600, 2)}


def truncation_counts(groups: dict[str, list[str]], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each model input limit, the longest input and the number over the limit, per group of PTB texts."""
    from src.metrics.preprocess import subword_lengths, truncation_summary

    rows = []
    for check in checks:
        for group, texts in groups.items():
            lengths = subword_lengths(texts, check["tokenizer"], check.get("suffix", ""))
            rows.append({"input": check["input"], "group": group, **truncation_summary(lengths, check["limit"])})
    return rows


def corpus_table(score_sets: list[ScoreSet], condition: str) -> dict[str, dict[str, float]]:
    """system -> metric -> corpus value, averaged over the condition's subsets."""
    table: dict[str, dict[str, list[float]]] = {}
    for score_set in score_sets:
        if score_set.condition == condition:
            for name, score in score_set.scores.items():
                table.setdefault(score_set.system, {}).setdefault(name, []).append(score.corpus)
    return {system: {name: float(np.mean(v)) for name, v in metrics.items()} for system, metrics in table.items()}


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def report_markdown(report: dict[str, Any]) -> str:
    determinism = report["determinism"]
    lines = [
        "# Metric implementation report (E11)",
        "",
        f"Generated by `scripts/validate_metrics.py` at commit `{report['run_info']['git_commit']}` "
        f"(dirty: {report['run_info']['git_dirty']}), {report['run_info']['timestamp_utc']}.",
        "",
        f"**Validation passed: {report['passed']}**",
        "",
        "## Environment and tools",
        "",
        f"- Platform `{report['tools']['platform']}`, Java `{report['tools']['java_version']}` "
        f"({report['tools']['jdk_release']}), device `{report['settings']['device']}`.",
        f"- `tools/MANIFEST.json`: {report['tools']['manifest_files']} files, "
        f"sha256 `{report['tools']['manifest_sha256']}`; all match: {report['tools']['manifest_ok']}.",
        "",
        "| Package | Version |",
        "|---|---|",
        *[f"| {name} | {version} |" for name, version in report["packages"].items()],
        "",
        "| Pinned model or file | Pin |",
        "|---|---|",
        *[f"| {name} | `{pin}` |" for name, pin in report["tools"]["pins"].items()],
        "",
        "## Metrics",
        "",
        f"Preprocessing: {report['preprocessing']}",
        "",
        "| Metric | Family | Corpus = mean of items | Needs audio |",
        "|---|---|---|---|",
        *[
            f"| {m['label']} (`{m['name']}`) | {m['family']} | {m['corpus_is_item_mean']} | {m['needs_audio']} |"
            for m in report["metrics"]
        ],
        "",
        "Parameters (`configs/metrics.yaml` `settings`):",
        "",
        *[f"- `{name}`: `{value}`" for name, value in report["settings"].items()],
        "",
        "## Integration tests",
        "",
        f"`make test-metrics`: {report['integration_tests']['summary']} "
        f"(exit {report['integration_tests']['returncode']}).",
        "",
        "## Validation run on the pilot",
        "",
        f"- {report['pilot']['recordings']} validation-split recordings, systems {report['pilot']['systems']}; "
        f"{report['pilot']['calls_per_run']} scoring calls per run (five_reference, four_reference drop_1-5, "
        "human_loo drop_1-5), all with audio.",
        f"- Caption normalization (E10 rule): {report['pilot']['normalization']}.",
        f"- Runs: {determinism['runs']}; bit-identical: **{determinism.get('identical', 'not checked')}** "
        f"({determinism.get('values_compared', 0)} values compared, max abs difference "
        f"{determinism.get('max_abs_difference')}).",
        "",
        "Pilot five-reference corpus scores (technical check on 20 recordings, not results):",
        "",
    ]
    table = report["pilot"]["five_reference"]
    names = [m.name for m in METRICS if m.family != "auxiliary" and any(m.name in v for v in table.values())]
    lines += [
        "| System | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        *[f"| {system} | " + " | ".join(_fmt(values[n]) for n in names) + " |" for system, values in table.items()],
        "",
        "## Truncation (inputs longer than a model's limit)",
        "",
        "| Input | Group | Limit | Longest | Over limit |",
        "|---|---|---|---|---|",
        *[
            f"| {r['input']} | {r['group']} | {r['limit']} | {r['max_length']} | {r['num_over_limit']} |"
            for r in report["truncation"]
        ],
        "",
        "## Timing (run 1, CPU)",
        "",
        f"Model loading: {report['timing']['load_models_seconds']:.1f} s.",
        "",
        "| Step | Pilot seconds | Calls | Full-set estimate (s) |",
        "|---|---|---|---|",
        *[
            f"| {name} | {seconds:.1f} | {report['timing']['calls'][name]} | "
            f"{report['timing']['extrapolation']['estimated_seconds'].get(name, '')} |"
            for name, seconds in report["timing"]["seconds"].items()
            if name != "load_models"
        ],
        "",
        f"Linear estimate for all conditions on {report['timing']['extrapolation']['full_items']} recordings: "
        f"**{report['timing']['extrapolation']['estimated_total_hours']} h** (upper bound; fixed per-call costs are "
        "scaled too).",
        "",
        f"Fixed + per-item estimate for the per-call steps, from one extra call on "
        f"{report['timing']['scaling']['large_items']} items: "
        f"**{report['timing']['scaling']['estimate']['estimated_hours']} h** over "
        f"{report['timing']['scaling']['estimate']['calls']} calls (the fluency detector and CLAP add the linear "
        "estimates above, as they embed each distinct text or recording once).",
        "",
        "| Step | Fixed s per call | s per item | Estimate (s) |",
        "|---|---|---|---|",
        *[
            f"| {name} | {step['fixed_seconds']} | {step['seconds_per_item']} | {step['estimated_seconds']} |"
            for name, step in report["timing"]["scaling"]["estimate"]["steps"].items()
        ],
        "",
        "## Interpretation and limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    return "\n".join(lines)
