"""Cross-metric and model comparison (E20).

Three analyses on the committed per-item tables, nothing rescored:

- Metric x metric Spearman correlations over captions (five references, E11), pooled over models and per model, with
  bootstrap CIs that resample recordings and keep each recording's captions together. Descriptive: with about a
  thousand recordings every correlation is far from zero, so no p-values are given.
- Model rankings by every metric (E11 item means), by the E17 share of captions in categories 1-3 and by the E18
  share rated accurate, with the share of bootstrap resamples that reproduce each observed order, and Kendall's tau-b
  between every pair of rankings (four models, so tau is coarse and untested).
- Within-recording concordance on the qualitative sample: for every pair of captions of the same recording with
  different E18 audio ratings, whether a metric scores the better-rated caption higher (1), equal (0.5) or lower (0).
  The share over all pairs is 0.5 for a metric unrelated to audio correctness. Recording and references are the same
  within a pair, so differences in recording difficulty cannot explain it. CIs resample recordings; the test flips
  the direction of every recording's pairs (sign flip of its summed concordance minus one half per pair), Holm across
  metrics.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np

from src.analysis.lengths import resampled_spearman, spearman_rows
from src.analysis.statistics import _num, holm, sign_flip_p
from src.evaluation.aggregate import bootstrap_indices, percentile_interval
from src.evaluation.scoring import ScoreSet

POOLED = "all"
CORRELATION_COLUMNS = (
    "scope",
    "metric_a",
    "metric_b",
    "num_recordings",
    "num_captions",
    "spearman",
    "ci_low",
    "ci_high",
)
RANKING_COLUMNS = ("ranker", "source", "system", "value", "rank", "share_observed_order")
TAU_COLUMNS = ("ranker_a", "ranker_b", "kendall_tau_b")
CONCORDANCE_COLUMNS = (
    "metric", "pairs", "num_pairs", "num_recordings", "concordant", "ties", "discordant", "concordance", "ci_low",
    "ci_high", "p_value", "p_holm", "significant",
)  # fmt: skip


def score_matrix(score_sets: list[ScoreSet], metric: str) -> np.ndarray:
    """(recordings, systems) item scores of one metric; every system scored on the same recordings in the same order."""
    first = score_sets[0]
    if any(s.sample_ids != first.sample_ids for s in score_sets):
        raise ValueError("Every system must be scored on the same recordings")
    return np.stack([np.asarray(s.scores[metric].items, dtype=np.float64) for s in score_sets], axis=1)


def metric_correlations(
    score_sets: list[ScoreSet], metrics: list[str], num_resamples: int, confidence: float, seed: int
) -> list[dict[str, Any]]:
    """Spearman's rho for every pair of metrics over captions, pooled over the systems and per system."""
    matrices = {metric: score_matrix(score_sets, metric) for metric in metrics}
    num_recordings = len(score_sets[0].sample_ids)
    indices = bootstrap_indices(num_recordings, num_resamples, seed)
    scopes = {POOLED: slice(None), **{s.system: slice(i, i + 1) for i, s in enumerate(score_sets)}}
    rows = []
    for scope, columns in scopes.items():
        for a, b in combinations(metrics, 2):
            x, y = matrices[a][:, columns], matrices[b][:, columns]
            resampled = resampled_spearman(x, y, indices)
            low, high = percentile_interval(resampled, confidence) if len(resampled) else (math.nan, math.nan)
            rows.append(
                {
                    "scope": scope,
                    "metric_a": a,
                    "metric_b": b,
                    "num_recordings": num_recordings,
                    "num_captions": x.size,
                    "spearman": _num(float(spearman_rows(x.ravel(), y.ravel())[0])),
                    "ci_low": _num(low),
                    "ci_high": _num(high),
                }
            )
    return rows


def ranks(values: dict[str, float]) -> dict[str, int]:
    """1 = highest value; ties broken by insertion order (system order)."""
    order = sorted(values, key=lambda system: -values[system])  # stable
    return {system: order.index(system) + 1 for system in values}


def order_stability(per_recording: np.ndarray, indices: np.ndarray) -> float:
    """Share of resamples of recordings in which the column means keep the observed order. `per_recording` is
    (recordings, systems); ties are broken by column order, as in E15."""
    observed = np.argsort(np.argsort(-per_recording.mean(axis=0), kind="stable"), kind="stable")
    resampled = per_recording[indices].mean(axis=1)  # (resamples, systems)
    order = np.argsort(np.argsort(-resampled, axis=1, kind="stable"), axis=1, kind="stable")
    return float((order == observed).all(axis=1).mean())


def kendall_tau_b(a: list[float], b: list[float]) -> float:
    """Kendall's tau-b between two equally long sequences (NaN if either is constant)."""
    concordant = discordant = ties_a = ties_b = 0
    for i, j in combinations(range(len(a)), 2):
        da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1
        elif db == 0:
            ties_b += 1
        elif da == db:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return (concordant - discordant) / denominator if denominator else math.nan


def ranking_rows(
    rankers: dict[str, dict[str, Any]], systems: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ranking table and Kendall's tau-b for every pair of rankers. `rankers` maps a ranker name to
    {"source": ..., "values": {system: value}, "stability": share}."""
    rows = []
    for name, ranker in rankers.items():
        values = {system: ranker["values"][system] for system in systems}
        for system, rank in ranks(values).items():
            rows.append(
                {
                    "ranker": name,
                    "source": ranker["source"],
                    "system": system,
                    "value": _num(values[system]),
                    "rank": rank,
                    "share_observed_order": _num(ranker["stability"]),
                }
            )
    taus = [
        {
            "ranker_a": a,
            "ranker_b": b,
            "kendall_tau_b": _num(
                kendall_tau_b([rankers[a]["values"][s] for s in systems], [rankers[b]["values"][s] for s in systems])
            ),
        }
        for a, b in combinations(rankers, 2)
    ]
    return rows, taus


def rating_pairs(ratings: np.ndarray) -> list[tuple[int, int]]:
    """(better, worse) caption positions of one recording: every pair with different ratings (1 is best)."""
    return [
        (i, j) if ratings[i] < ratings[j] else (j, i)
        for i, j in combinations(range(len(ratings)), 2)
        if ratings[i] != ratings[j]
    ]


def concordance_rows(
    ratings: np.ndarray,
    scores: dict[str, np.ndarray],
    num_resamples: int,
    num_permutations: int,
    confidence: float,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Within-recording concordance per metric, for all pairs with different ratings and for pairs rated 1 vs 3.
    `ratings` and every `scores[metric]` are (recordings, captions per recording)."""
    selections = {"all": lambda r_best, r_worst: True, "1_vs_3": lambda r_best, r_worst: (r_best, r_worst) == (1, 3)}
    num_recordings = ratings.shape[0]
    indices = bootstrap_indices(num_recordings, num_resamples, seed)
    flips = 2 * np.random.default_rng(seed).integers(0, 2, size=(num_permutations, num_recordings)) - 1.0
    rows = []
    for name, keep in selections.items():
        pairs = [
            [(i, j) for i, j in rating_pairs(ratings[r]) if keep(ratings[r][i], ratings[r][j])]
            for r in range(num_recordings)
        ]
        counts = np.array([len(p) for p in pairs], dtype=np.float64)
        family = []
        for metric, values in scores.items():
            outcome = [[np.sign(values[r][i] - values[r][j]) for i, j in pairs[r]] for r in range(num_recordings)]
            concordant = np.array([sum(o > 0 for o in row) for row in outcome], dtype=np.float64)
            tied = np.array([sum(o == 0 for o in row) for row in outcome], dtype=np.float64)
            credit = concordant + 0.5 * tied  # per recording
            share = credit.sum() / counts.sum()
            resampled = credit[indices].sum(axis=1) / counts[indices].sum(axis=1)
            low, high = percentile_interval(resampled[np.isfinite(resampled)], confidence)
            centred = credit - 0.5 * counts  # 0 in expectation when the metric ignores the ratings
            used = counts > 0
            p = sign_flip_p(centred[used], flips[:, used])
            family.append(
                {
                    "metric": metric,
                    "pairs": name,
                    "num_pairs": int(counts.sum()),
                    "num_recordings": int(used.sum()),
                    "concordant": int(concordant.sum()),
                    "ties": int(tied.sum()),
                    "discordant": int(counts.sum() - concordant.sum() - tied.sum()),
                    "concordance": _num(share),
                    "ci_low": _num(low),
                    "ci_high": _num(high),
                    "p_value": _num(p),
                }
            )
        for row, adjusted in zip(family, holm([float(r["p_value"]) for r in family]), strict=True):
            row["p_holm"] = _num(adjusted)
            row["significant"] = bool(adjusted < alpha)
        rows += family
    return rows


LIMITATIONS = [
    "The E17 and E18 rankings come from the 50-recording qualitative sample and one reviewer; the metric rankings from "
    "all 1,045 recordings. Their order-stability shares resample different numbers of recordings.",
    "Kendall's tau over four models takes only seven values and is not tested; it summarises agreement between "
    "rankings, not its significance.",
    "Concordance uses the 200 sampled captions: pairs within a recording share the recording and the references, but "
    "the pairs of one recording are not independent (the CI and test resample whole recordings). E18 ratings are one "
    "reviewer's judgements.",
    "Metric correlations are over captions scored against five references (E11); they describe how the metrics vary "
    "together, not which metric is right.",
]


def _f(value: str, digits: int = 2) -> str:
    return "n/a" if value in ("", None) or math.isnan(float(value)) else f"{float(value):.{digits}f}"


def _p(value: str) -> str:
    return "n/a" if value == "" else ("< 0.001" if float(value) < 0.001 else f"{float(value):.3f}")


def report_markdown(report: dict[str, Any]) -> str:
    names, labels, run_info = report["display_names"], report["ranker_labels"], report["run_info"]
    systems, metrics = list(names), report["metrics"]
    lines = [
        "# Cross-metric and model comparison (E20)",
        "",
        f"Generated by `scripts/compare_metrics.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        "## Inputs",
        "",
        *[f"- `{item['path']}` (sha256 `{item['sha256']}`)" for item in report["inputs"].values()],
        "",
        "## Model rankings",
        "",
        "Value and rank (1 = best) of each model. Metrics: item mean over the 1,045 recordings, five references (E11); "
        "stability is the share of 10,000 bootstrap resamples reproducing the observed order (E15). Review rows: share "
        "of the model's 50 sampled captions in E17 categories 1-3 (agreement with the references' main content) or "
        "rated accurate in E18; stability resamples the 50 recordings.",
        "",
        "| Ranking | " + " | ".join(names[s] for s in systems) + " | Stability | tau vs E17 | tau vs E18 |",
        "|---|" + "---|" * (len(systems) + 3),
    ]
    table = {(r["ranker"], r["system"]): r for r in report["rankings"]}
    tau = {frozenset((r["ranker_a"], r["ranker_b"])): r["kendall_tau_b"] for r in report["kendall_tau"]}
    e18, e19 = report["review_rankers"]
    for ranker in labels:
        cells = [f"{_f(table[(ranker, s)]['value'], 3)} ({table[(ranker, s)]['rank']})" for s in systems]
        taus = [_f(tau[frozenset((ranker, other))]) if ranker != other else "-" for other in (e18, e19)]
        stability = f"{float(table[(ranker, systems[0])]['share_observed_order']):.0%}"
        lines.append(f"| {labels[ranker]} | " + " | ".join(cells) + f" | {stability} | " + " | ".join(taus) + " |")
    lines += [
        "",
        f"Kendall's tau-b for every pair of rankings is in `{report['outputs']['kendall_tau']}`.",
        "",
        "## Within-recording concordance with the audio ratings",
        "",
        "Pairs of sampled captions of the same recording with different E18 ratings; concordance is the share in which "
        "the metric scores the better-rated caption higher (ties count one half; 0.5 = unrelated to the ratings). "
        f"{report['confidence']:.0%} bootstrap CI over recordings; sign-flip test over recordings, Holm across the "
        f"{len(metrics)} metrics within each set of pairs.",
        "",
    ]
    for pairs, title in (("all", "All pairs with different ratings"), ("1_vs_3", "Pairs rated 1 vs 3")):
        rows = [r for r in report["concordance"] if r["pairs"] == pairs]
        lines += [
            f"**{title}** ({rows[0]['num_pairs']} pairs from {rows[0]['num_recordings']} recordings)",
            "",
            "| Metric | Concordant / tied / discordant | Concordance [CI] | p (Holm) |",
            "|---|---|---|---|",
            *[
                f"| {report['metric_labels'][r['metric']]} | {r['concordant']} / {r['ties']} / {r['discordant']} | "
                f"{_f(r['concordance'])} [{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {_p(r['p_holm'])}"
                f"{' *' if r['significant'] else ''} |"
                for r in rows
            ],
            "",
        ]
    corr = {(r["scope"], r["metric_a"], r["metric_b"]): r for r in report["correlations"]}

    def rho(scope: str, a: str, b: str) -> dict[str, Any]:
        return corr.get((scope, a, b)) or corr[(scope, b, a)]

    short = report["metric_labels"]
    lines += [
        "## Metric correlations",
        "",
        "Spearman's rho between metrics over captions (five references, E11). Lower triangle: pooled over the four "
        "models (4,180 captions). Upper triangle: smallest and largest rho over the four models separately (1,045 "
        f"captions each). {report['confidence']:.0%} bootstrap CIs over recordings are in "
        f"`{report['outputs']['correlations']}`.",
        "",
        "| | " + " | ".join(short[m] for m in metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for i, a in enumerate(metrics):
        cells = []
        for j, b in enumerate(metrics):
            if i == j:
                cells.append("-")
            elif j < i:
                cells.append(_f(rho(POOLED, a, b)["spearman"]))
            else:
                values = [float(rho(s, a, b)["spearman"]) for s in systems]
                cells.append(f"{min(values):.2f}-{max(values):.2f}")
        lines.append(f"| {short[a]} | " + " | ".join(cells) + " |")
    lines += ["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""]
    lines += ["## Outputs", "", *[f"- `{path}`" for path in report["outputs"].values()], ""]
    return "\n".join(lines)
