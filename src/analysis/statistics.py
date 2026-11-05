"""Formal tests, multiple-comparison correction, effect sizes and ranking stability (E16).

Every comparison is paired over recordings: two systems scored on the same recordings with the same references. The
test is a sign-flip permutation test of the mean per-recording difference (two-sided; flipping a sign swaps the two
systems' scores within that recording), with p = (1 + #{|permuted mean| >= |observed mean|}) / (1 + permutations).
Holm-Bonferroni is applied within each family on its primary sample (every recording); sensitivity samples keep
uncorrected p-values. CIs are percentile bootstrap intervals over recordings, effect sizes are paired d_z (mean
difference / SD of the differences). Length-score correlations are tested by permuting the scores across recordings,
each recording keeping its captions together. Nothing is rescored: the inputs are the committed per-item tables.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.aggregate import bootstrap_indices, bootstrap_mean
from src.evaluation.scoring import ScoreSet
from src.metrics.registry import BY_NAME, METRICS

MAIN_METRICS = [metric.name for metric in METRICS if metric.family != "auxiliary"]  # all higher-is-better

TEST_COLUMNS = (
    "family", "condition", "subset", "sample", "num_items", "metric", "system_a", "system_b", "difference", "ci_low",
    "ci_high", "effect_size", "p_value", "p_holm", "significant", "significant_uncorrected",
)  # fmt: skip
CORRELATION_COLUMNS = (
    "family", "sample", "system", "predictor", "metric", "num_recordings", "num_pairs", "spearman", "ci_low", "ci_high",
    "p_value", "p_holm", "significant", "significant_uncorrected",
)  # fmt: skip
RANK_COLUMNS = ("condition", "subset", "sample", "metric", "system", "item_mean", "observed_rank", "rank", "share")
ORDER_COLUMNS = ("condition", "subset", "sample", "metric", "observed_order", "share_observed_order", "num_resamples")


def _num(value: float) -> str:
    return "" if value is None or math.isnan(value) else repr(float(value))


@dataclass
class Resampling:
    """Bootstrap indices, sign flips and permutations, one draw per number of recordings (shared by all systems)."""

    num_resamples: int
    num_permutations: int
    seed: int
    confidence: float
    alpha: float
    _cache: dict[tuple[str, int], np.ndarray] = field(default_factory=dict, repr=False)

    def indices(self, num_items: int) -> np.ndarray:
        key = ("indices", num_items)
        if key not in self._cache:
            self._cache[key] = bootstrap_indices(num_items, self.num_resamples, self.seed)
        return self._cache[key]

    def signs(self, num_items: int) -> np.ndarray:
        """(permutations, items) of +1/-1."""
        key = ("signs", num_items)
        if key not in self._cache:
            rng = np.random.default_rng(self.seed)
            flips = rng.integers(0, 2, size=(self.num_permutations, num_items), dtype=np.int8)
            self._cache[key] = (2 * flips - 1).astype(np.float64)
        return self._cache[key]

    def permutations(self, num_items: int) -> np.ndarray:
        """(permutations, items) of recording orders."""
        key = ("permutations", num_items)
        if key not in self._cache:
            rng = np.random.default_rng(self.seed)
            self._cache[key] = rng.permuted(np.tile(np.arange(num_items), (self.num_permutations, 1)), axis=1)
        return self._cache[key]


def _p_value(observed: float, permuted: np.ndarray) -> float:
    """Two-sided permutation p-value; a small tolerance keeps ties from floating-point noise counted."""
    tolerance = 1e-12 * max(1.0, abs(observed))
    return float((1 + np.count_nonzero(np.abs(permuted) >= abs(observed) - tolerance)) / (1 + len(permuted)))


def sign_flip_p(differences: np.ndarray, signs: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    return _p_value(float(differences.mean()), signs @ differences / len(differences))


def effect_size(differences: np.ndarray) -> float:
    """Paired d_z: mean difference / SD of the differences (NaN when every difference is equal)."""
    differences = np.asarray(differences, dtype=np.float64)
    sd = differences.std(ddof=1) if len(differences) > 1 else 0.0
    return float(differences.mean() / sd) if sd > 0 else math.nan


def holm(p_values: list[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1), in input order."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1.0)
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def paired_test(a: np.ndarray, b: np.ndarray, resampling: Resampling) -> dict[str, Any]:
    differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ci = bootstrap_mean(differences, resampling.indices(len(differences)), resampling.confidence)
    p = sign_flip_p(differences, resampling.signs(len(differences)))
    return {
        "num_items": len(differences),
        "difference": _num(ci["mean"]),
        "ci_low": _num(ci["ci_low"]),
        "ci_high": _num(ci["ci_high"]),
        "effect_size": _num(effect_size(differences)),
        "p_value": _num(p),
        "p_holm": "",
        "significant": "",
        "significant_uncorrected": p < resampling.alpha,
    }


def comparison_rows(
    family: str, sample: str, pairs: list[tuple[ScoreSet, ScoreSet]], metrics: list[str], resampling: Resampling
) -> list[dict[str, Any]]:
    """One test row per metric and pair (system_a minus system_b), metrics outermost."""
    rows = []
    for metric in metrics:
        for a, b in pairs:
            if a.sample_ids != b.sample_ids:
                raise ValueError(f"{a.system} and {b.system} are not scored on the same recordings ({sample})")
            test = paired_test(a.scores[metric].items, b.scores[metric].items, resampling)
            rows.append(
                {
                    "family": family,
                    "condition": a.condition,
                    "subset": a.subset,
                    "sample": sample,
                    "metric": metric,
                    "system_a": a.system,
                    "system_b": b.system,
                    **test,
                }
            )
    return [{column: row[column] for column in TEST_COLUMNS} for row in rows]


def model_pairs(score_sets: list[ScoreSet]) -> list[tuple[ScoreSet, ScoreSet]]:
    return list(combinations(score_sets, 2))


def apply_holm(rows: list[dict[str, Any]], primary: Callable[[dict[str, Any]], bool], alpha: float) -> None:
    """Holm-adjust, per family, the rows selected by `primary`; the others keep only their uncorrected p-value."""
    for family in dict.fromkeys(row["family"] for row in rows):
        group = [row for row in rows if row["family"] == family and primary(row)]
        if not group:
            continue
        for row, adjusted in zip(group, holm([float(row["p_value"]) for row in group]), strict=True):
            row["p_holm"] = _num(adjusted)
            row["significant"] = bool(adjusted < alpha)


def permutation_spearman_p(x: np.ndarray, y: np.ndarray, permutations: np.ndarray, chunk: int = 1000) -> float:
    """Two-sided p-value of Spearman's rho between x and y, both (recordings, captions per recording), permuting the
    recordings of y. Pooled ranks do not change under the permutation, so they are computed once."""
    if x.shape != y.shape:
        raise ValueError(f"Shapes differ: {x.shape} vs {y.shape}")
    rx = pd.Series(x.ravel()).rank().to_numpy().reshape(x.shape)
    ry = pd.Series(y.ravel()).rank().to_numpy().reshape(y.shape)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    norm = math.sqrt(float((rx**2).sum() * (ry**2).sum()))
    if norm == 0:
        return math.nan
    observed = float((rx * ry).sum()) / norm
    permuted = np.concatenate(
        [
            (ry[part] * rx[None]).sum(axis=(1, 2)) / norm
            for part in (permutations[start : start + chunk] for start in range(0, len(permutations), chunk))
        ]
    )
    return _p_value(observed, permuted)


def correlation_rows(
    family: str,
    associations: list[dict[str, Any]],
    data: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]],
    resampling: Resampling,
) -> list[dict[str, Any]]:
    """E15-style association rows (Spearman and its bootstrap CI) with permutation p-values; `data` maps (system,
    sample, predictor, metric) to the (x, y) arrays the row was computed from."""
    rows = []
    for row in associations:
        x, y = data[(row["system"], row["sample"], row["predictor"], row["metric"])]
        p = permutation_spearman_p(x, y, resampling.permutations(x.shape[0]))
        rows.append(
            {
                **{column: row[column] for column in CORRELATION_COLUMNS if column in row},
                "family": family,
                "p_value": _num(p),
                "p_holm": "",
                "significant": "",
                "significant_uncorrected": bool(p < resampling.alpha),
            }
        )
    return [{column: row[column] for column in CORRELATION_COLUMNS} for row in rows]


def rank_tables(
    score_sets: list[ScoreSet], sample: str, metrics: list[str], resampling: Resampling
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per metric: each system's share of bootstrap resamples at each rank (1 = highest mean), and the share of
    resamples that reproduce the observed order. Ties are broken by system order."""
    first = score_sets[0]
    if any(s.sample_ids != first.sample_ids for s in score_sets):
        raise ValueError("Every system must be scored on the same recordings")
    indices = resampling.indices(len(first.sample_ids))
    systems = [s.system for s in score_sets]
    rank_rows, order_rows = [], []
    for metric in metrics:
        if BY_NAME[metric].family == "auxiliary":
            raise ValueError(f"{metric}: ranks are defined for the main (higher-is-better) metrics only")
        observed = np.array([s.scores[metric].items.mean() for s in score_sets])
        resampled = np.stack([s.scores[metric].items[indices].mean(axis=1) for s in score_sets])
        observed_rank = np.argsort(np.argsort(-observed, kind="stable"), kind="stable") + 1
        ranks = np.argsort(np.argsort(-resampled, axis=0, kind="stable"), axis=0, kind="stable") + 1
        for index, system in enumerate(systems):
            for rank in range(1, len(systems) + 1):
                rank_rows.append(
                    {
                        "condition": first.condition,
                        "subset": first.subset,
                        "sample": sample,
                        "metric": metric,
                        "system": system,
                        "item_mean": _num(observed[index]),
                        "observed_rank": int(observed_rank[index]),
                        "rank": rank,
                        "share": _num(float((ranks[index] == rank).mean())),
                    }
                )
        order_rows.append(
            {
                "condition": first.condition,
                "subset": first.subset,
                "sample": sample,
                "metric": metric,
                "observed_order": ";".join(systems[i] for i in np.argsort(observed_rank)),
                "share_observed_order": _num(float((ranks == observed_rank[:, None]).all(axis=0).mean())),
                "num_resamples": len(indices),
            }
        )
    return rank_rows, order_rows


def decision_changes(
    tests: list[dict[str, Any]], earlier: list[dict[str, str]], family: str, sample: str
) -> list[dict[str, Any]]:
    """Comparisons whose Holm decision differs from the earlier task's uncorrected 1,000-resample CI mark."""
    marks = {(r["metric"], r["system_a"], r["system_b"]): r["excludes_zero"] == "True" for r in earlier}
    changes = []
    for row in tests:
        if row["family"] != family or row["sample"] != sample:
            continue
        before = marks[(row["metric"], row["system_a"], row["system_b"])]
        if before != row["significant"]:
            changes.append({**{k: row[k] for k in ("metric", "system_a", "system_b")}, "before": before, "row": row})
    return changes


def sensitivity_counts(rows: list[dict[str, Any]], primary_sample: str, key: Callable) -> list[dict[str, Any]]:
    """Per family and sensitivity sample: uncorrected significant count, and the comparisons whose uncorrected
    decision differs from the Holm decision on the primary sample."""
    primary = {(row["family"], key(row)): row for row in rows if row["sample"] == primary_sample}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["sample"] != primary_sample:
            groups.setdefault((row["family"], row["sample"]), []).append(row)
    out = []
    for (family, sample), group in groups.items():
        differs = [row for row in group if row["significant_uncorrected"] != primary[(family, key(row))]["significant"]]
        out.append(
            {
                "family": family,
                "sample": sample,
                "num_tests": len(group),
                "significant_uncorrected": sum(bool(row["significant_uncorrected"]) for row in group),
                "differs_from_primary": [key(row) for row in differs],
            }
        )
    return out


def _fmt(value: str, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}" if value != "" else "n/a"


def _p(value: str) -> str:
    if value == "":
        return "n/a"
    p = float(value)
    return f"{p:.4f}" if p >= 0.001 else f"{p:.1e}"


ASSUMPTIONS = [
    "The unit of analysis is the recording. Recordings are treated as exchangeable draws from the population Clotho's "
    "evaluation split represents; captions of the same recording are never treated as independent (human captions are "
    "averaged over the five held-out positions, or resampled together for correlations).",
    "Every comparison is paired: both systems are scored on the same recordings with the same references, metric "
    "implementations and aggregation. Five-reference (E12) and four-reference (E13-E14) scores are never compared "
    "with each other, because CIDEr-D, SPIDEr and SPIDEr-FL depend on the reference set.",
    "The tested statistic is the mean of per-recording scores. For BLEU and METEOR the published corpus value comes "
    "from corpus n-gram statistics and is reported separately; their tests and CIs are on the per-item mean (D5).",
    "The sign-flip test assumes that, under the null hypothesis, the two systems' scores of a recording are "
    "exchangeable; it tests the mean difference, not a difference in distribution. Bootstrap CIs are percentile "
    "intervals, which can be slightly narrow for skewed metrics such as CIDEr-D.",
    "Multiple comparisons: Holm-Bonferroni within each family, on every recording only (the families are fixed in "
    "configs/metrics.yaml before testing). Sensitivity samples and the second length predictor are uncorrected checks.",
    "Effect sizes (paired d_z) are reported because with 1,045 recordings very small differences, such as raw "
    "BERTScore-F1 gaps of about 0.002, can be significant without being meaningful.",
    "API models (GPT-Audio-1.5, Gemini 3.8 Flash) are not reproducible at temperature 0 (E05); every test is "
    "conditional on the frozen captions and does not include run-to-run variation.",
]


def report_markdown(report: dict[str, Any]) -> str:
    """The statistical report: numbers, IDs and settings only, never caption text."""
    names = {**report["display_names"], "human": "Human", "human_reference_mean": "Human reference mean"}
    run_info, settings, alpha = report["run_info"], report["settings"], report["settings"]["alpha"]
    tests, primary = report["tests"], report["primary_sample"]
    lines = [
        "# Statistical comparison (E16)",
        "",
        f"Generated by `scripts/analyze_results.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        "## Settings",
        "",
        f"- Sign-flip permutation tests with {settings['num_permutations']:,} permutations; bootstrap CIs with "
        f"{settings['num_resamples']:,} resamples of recordings ({settings['confidence']:.0%} percentile); "
        f"seed {settings['seed']}; alpha {alpha}.",
        "- Holm-Bonferroni within each family on every recording:",
        *[f"  - `{family}`: {count} tests. {report['families'][family]}" for family, count in report["counts"].items()],
        "- `*` marks a Holm-adjusted p below alpha. Full p-values, adjusted p-values and effect sizes are in the CSVs.",
        "",
        "Inputs:",
        "",
        *[f"- `{item['path']}` (sha256 `{item['sha256']}`)" for item in report["inputs"].values()],
        "",
        "## Assumptions and procedures",
        "",
        *[f"- {item}" for item in report["assumptions"]],
        "",
    ]

    def cell(row: dict[str, Any]) -> str:
        mark = "*" if row["significant"] else ""
        return f"{_fmt(row['difference'])}{mark} (d={float(row['effect_size']):+.2f})" if row["effect_size"] else "n/a"

    for family, title, note in (
        ("rq1_model_pairs", "RQ1: paired differences between models (five references, E12)", "row minus column"),
        ("rq2_model_vs_human", "RQ2: models minus human (matched four references, E14)", "model minus human"),
    ):
        rows = [r for r in tests if r["family"] == family and r["sample"] == primary]
        pairs = list(dict.fromkeys((r["system_a"], r["system_b"]) for r in rows))
        by = {(r["metric"], r["system_a"], r["system_b"]): r for r in rows}
        header = [f"{names[a]} - {names[b]}" if family == "rq1_model_pairs" else names[a] for a, b in pairs]
        lines += [
            f"## {title}",
            "",
            f"Item-mean difference ({note}), `*` Holm-significant, paired d_z in parentheses.",
            "",
            "| Metric | " + " | ".join(header) + " |",
            "|---|" + "---|" * len(pairs),
        ]
        for metric in report["metrics"]:
            cells = [cell(by[(metric, a, b)]) for a, b in pairs]
            lines.append(f"| {BY_NAME[metric].label} | " + " | ".join(cells) + " |")
        significant = sum(bool(r["significant"]) for r in rows)
        changes = report["decision_changes"][family]
        lines += [
            "",
            f"{significant} of {len(rows)} comparisons are significant after correction. Compared with the uncorrected "
            f"1,000-resample CI marks of {report['earlier'][family]}, {len(changes)} decisions change:",
            "",
            *(
                [
                    f"- {BY_NAME[c['metric']].label}, {names[c['system_a']]} - {names[c['system_b']]}: "
                    f"{'significant' if c['before'] else 'not significant'} before, "
                    f"{'significant' if c['row']['significant'] else 'not significant'} now "
                    f"(p = {_p(c['row']['p_value'])}, Holm p = {_p(c['row']['p_holm'])})"
                    for c in changes
                ]
                or ["- none"]
            ),
            "",
        ]
    lines += [
        "## Ranking stability",
        "",
        "Share of bootstrap resamples that reproduce the observed order of item means (highest first), and each "
        "system's share at its observed rank.",
        "",
    ]
    for condition, subset, title in report["rank_titles"]:
        key = (condition, subset, primary)
        orders = [r for r in report["orders"] if (r["condition"], r["subset"], r["sample"]) == key]
        at_rank = {
            (r["metric"], r["system"]): float(r["share"])
            for r in report["ranks"]
            if (r["condition"], r["subset"], r["sample"]) == key and r["rank"] == r["observed_rank"]
        }
        lines += [f"### {title}", "", "| Metric | Observed order (share at observed rank) | Same order |"]
        lines.append("|---|---|---|")
        for row in orders:
            systems = row["observed_order"].split(";")
            order = " > ".join(f"{names[s]} ({at_rank[(row['metric'], s)]:.0%})" for s in systems)
            lines.append(f"| {BY_NAME[row['metric']].label} | {order} | {float(row['share_observed_order']):.1%} |")
        lines.append("")
    length = [r for r in tests if r["family"] == "length_vs_human" and r["sample"] == primary]
    lines += [
        "## Caption length (E15)",
        "",
        "Model caption length minus the mean length of its recording's five references (words).",
        "",
        "| Model | Difference [95% CI] | d_z | p | Holm p |",
        "|---|---|---|---|---|",
        *[
            f"| {names[r['system_a']]} | {_fmt(r['difference'], 2)}{'*' if r['significant'] else ''} "
            f"[{_fmt(r['ci_low'], 2)}, {_fmt(r['ci_high'], 2)}] | {float(r['effect_size']):+.2f} | {_p(r['p_value'])} "
            f"| {_p(r['p_holm'])} |"
            for r in length
        ],
        "",
        "Spearman correlation between caption length and score (raw length; `*` Holm-significant within the family).",
        "",
    ]
    corr = {
        (r["system"], r["metric"]): r
        for r in report["correlations"]
        if r["sample"] == primary and r["predictor"] == "words"
    }
    systems = list(dict.fromkeys(r["system"] for r in report["correlations"]))
    lines += ["| Metric | " + " | ".join(names[s] for s in systems) + " |", "|---|" + "---|" * len(systems)]
    for metric in report["association_metrics"]:
        cells = [
            f"{_fmt(corr[(s, metric)]['spearman'], 2)}{'*' if corr[(s, metric)]['significant'] else ''}"
            for s in systems
        ]
        lines.append(f"| {BY_NAME[metric].label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Sensitivity samples (uncorrected)",
        "",
        "Comparisons whose uncorrected decision (p < alpha) on a sensitivity sample differs from the Holm decision on "
        "every recording.",
        "",
        "| Family | Sample | Tests | Significant (uncorrected) | Differ from every-recording decision |",
        "|---|---|---|---|---|",
    ]
    for row in report["sensitivity"]:
        differs = row["differs_from_primary"]
        listed = "; ".join(" ".join(str(part) for part in item) for item in differs) if len(differs) <= 8 else ""
        detail = f"{len(differs)}" + (f": {listed}" if listed else "")
        counts = f"{row['num_tests']} | {row['significant_uncorrected']}"
        lines.append(f"| `{row['family']}` | `{row['sample']}` | {counts} | {detail} |")
    lines += ["", "## Outputs", "", *[f"- `{path}`" for path in report["outputs"].values()], ""]
    return "\n".join(lines)
