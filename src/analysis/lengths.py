"""Caption-length statistics, model-human length comparison and length-score associations (E15).

Words are the whitespace tokens of the scored caption (`caption.split()`), the definition of the dataset and caption
completion reports. Each model has one caption per recording; the human references have five, one per CSV position.
A model caption is compared with the mean length of its recording's five references; a held-out human caption with the
mean of the other four (the references it is scored against in E13). Bootstrap resamples are over recordings with the
indices of E12-E14 (same count and seed), so a recording's five human captions are resampled together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.aggregate import bootstrap_indices, bootstrap_mean, percentile_interval
from src.metrics.registry import BY_NAME

HUMAN = "human"
PREDICTORS = ("words", "words_minus_reference_mean")

PER_CAPTION_COLUMNS = ("system", "sample_id", "position", "words", "reference_mean_words", "reference_max_words")
SUMMARY_COLUMNS = (
    "system", "sample", "num_recordings", "num_captions", "mean", "sd", "min", "q1", "median", "q3", "iqr", "max",
    "ratio_to_human", "ratio_ci_low", "ratio_ci_high", "difference_mean", "difference_ci_low", "difference_ci_high",
    "share_longer_than_reference_mean", "share_longer_than_all_references",
)  # fmt: skip
ASSOCIATION_COLUMNS = (
    "system", "sample", "predictor", "metric", "num_recordings", "num_pairs", "spearman", "ci_low", "ci_high",
    "excludes_zero",
)  # fmt: skip


def word_count(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Lengths:
    """Word counts of one system, shape (recordings, captions per recording), and of the references each is compared
    with (their mean and maximum)."""

    system: str
    sample_ids: list[str]
    words: np.ndarray
    reference_mean: np.ndarray
    reference_max: np.ndarray

    def restrict(self, sample_ids: list[str]) -> Lengths:
        index = {sid: i for i, sid in enumerate(self.sample_ids)}
        if missing := [sid for sid in sample_ids if sid not in index]:
            raise ValueError(f"{self.system}: no lengths for {len(missing)} recordings, e.g. {missing[:3]}")
        rows = [index[sid] for sid in sample_ids]
        return Lengths(
            self.system, list(sample_ids), self.words[rows], self.reference_mean[rows], self.reference_max[rows]
        )


def model_lengths(
    system: str, captions: dict[str, str], references: dict[str, list[str]], sample_ids: list[str]
) -> Lengths:
    words = np.array([[word_count(captions[sid])] for sid in sample_ids], dtype=np.float64)
    refs = [[word_count(ref) for ref in references[sid]] for sid in sample_ids]
    mean = np.array([[np.mean(r)] for r in refs], dtype=np.float64)
    top = np.array([[max(r)] for r in refs], dtype=np.float64)
    return Lengths(system, list(sample_ids), words, mean, top)


def human_lengths(references: dict[str, list[str]], sample_ids: list[str]) -> Lengths:
    """Every reference as a held-out caption, compared with the other references of its recording."""
    words = np.array([[word_count(ref) for ref in references[sid]] for sid in sample_ids], dtype=np.float64)
    if words.ndim != 2 or words.shape[1] < 2:
        raise ValueError("Every recording needs the same number (at least 2) of references")
    others = [np.delete(words, k, axis=1) for k in range(words.shape[1])]
    mean = np.stack([o.mean(axis=1) for o in others], axis=1)
    top = np.stack([o.max(axis=1) for o in others], axis=1)
    return Lengths(HUMAN, list(sample_ids), words, mean, top)


def per_caption_rows(lengths: list[Lengths]) -> list[dict[str, Any]]:
    """One row per caption; `position` is the CSV position (1-5) for human captions and empty for model captions."""
    rows = []
    for item in lengths:
        human = item.system == HUMAN
        for i, sid in enumerate(item.sample_ids):
            for k in range(item.words.shape[1]):
                rows.append(
                    {
                        "system": item.system,
                        "sample_id": sid,
                        "position": k + 1 if human else "",
                        "words": int(item.words[i, k]),
                        "reference_mean_words": repr(float(item.reference_mean[i, k])),
                        "reference_max_words": int(item.reference_max[i, k]),
                    }
                )
    return rows


def _num(value: float) -> str:
    return "" if value is None or math.isnan(value) else repr(float(value))


def length_stats(words: np.ndarray) -> dict[str, float]:
    values = np.asarray(words, dtype=np.float64).ravel()
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
        "min": float(values.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "max": float(values.max()),
    }


def summary_rows(
    lengths: list[Lengths], sample: str, num_resamples: int, confidence: float, seed: int
) -> list[dict[str, Any]]:
    """Length statistics per system; for models also the ratio of mean lengths to the human mean and the paired
    per-recording difference from the reference mean, each with a bootstrap CI over recordings."""
    (human,) = [item for item in lengths if item.system == HUMAN]
    indices = bootstrap_indices(len(human.sample_ids), num_resamples, seed)
    human_mean = human.words.mean(axis=1)
    rows = []
    for item in lengths:
        if item.sample_ids != human.sample_ids:
            raise ValueError(f"{item.system} covers different recordings from the human references")
        row: dict[str, Any] = {
            "system": item.system,
            "sample": sample,
            "num_recordings": len(item.sample_ids),
            "num_captions": item.words.size,
            **{key: _num(value) for key, value in length_stats(item.words).items()},
            "share_longer_than_reference_mean": _num(float((item.words > item.reference_mean).mean())),
            "share_longer_than_all_references": _num(float((item.words > item.reference_max).mean())),
        }
        row |= dict.fromkeys(("ratio_to_human", "ratio_ci_low", "ratio_ci_high"), "")
        row |= dict.fromkeys(("difference_mean", "difference_ci_low", "difference_ci_high"), "")
        if item.system != HUMAN:
            words = item.words[:, 0]
            ratios = words[indices].mean(axis=1) / human_mean[indices].mean(axis=1)
            low, high = percentile_interval(ratios, confidence)
            row |= {"ratio_to_human": _num(words.mean() / human_mean.mean()), "ratio_ci_low": _num(low)}
            row["ratio_ci_high"] = _num(high)
            difference = bootstrap_mean(words - item.reference_mean[:, 0], indices, confidence)
            row |= {"difference_mean": _num(difference["mean"]), "difference_ci_low": _num(difference["ci_low"])}
            row["difference_ci_high"] = _num(difference["ci_high"])
        rows.append({column: row[column] for column in SUMMARY_COLUMNS})
    return rows


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman's rho of each row of x with the same row of y (average ranks for ties); NaN for a constant row."""
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    rx = pd.DataFrame(x).rank(axis=1).to_numpy(dtype=np.float64, copy=True)
    ry = pd.DataFrame(y).rank(axis=1).to_numpy(dtype=np.float64, copy=True)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(axis=1) / np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))


def resampled_spearman(x: np.ndarray, y: np.ndarray, indices: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Spearman's rho on each resample of recordings (rows of x and y, each with all its captions); finite values
    only. Resamples are processed in chunks to bound memory."""
    num_resamples = len(indices)
    values = [
        spearman_rows(x[part].reshape(len(part), -1), y[part].reshape(len(part), -1))
        for part in (indices[start : start + chunk] for start in range(0, num_resamples, chunk))
    ]
    resampled = np.concatenate(values)
    return resampled[np.isfinite(resampled)]


def association_rows(
    lengths: Lengths, scores: dict[str, np.ndarray], sample: str, num_resamples: int, confidence: float, seed: int
) -> list[dict[str, Any]]:
    """Spearman's rho between caption length (or its difference from the reference mean) and the per-caption score,
    for each metric in `scores` (arrays shaped like `lengths.words`). The CI resamples recordings, keeping all the
    captions of a recording together."""
    num_recordings = len(lengths.sample_ids)
    indices = bootstrap_indices(num_recordings, num_resamples, seed)
    predictors = {"words": lengths.words, "words_minus_reference_mean": lengths.words - lengths.reference_mean}
    rows = []
    for metric, values in scores.items():
        if values.shape != lengths.words.shape:
            raise ValueError(f"{lengths.system} {metric}: scores {values.shape} vs lengths {lengths.words.shape}")
        for predictor, x in predictors.items():
            rho = float(spearman_rows(x.ravel(), values.ravel())[0])
            resampled = resampled_spearman(x, values, indices)
            low, high = percentile_interval(resampled, confidence) if len(resampled) else (math.nan, math.nan)
            rows.append(
                {
                    "system": lengths.system,
                    "sample": sample,
                    "predictor": predictor,
                    "metric": metric,
                    "num_recordings": num_recordings,
                    "num_pairs": x.size,
                    "spearman": _num(rho),
                    "ci_low": _num(low),
                    "ci_high": _num(high),
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return rows


def _fmt(value: str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}" if value != "" else "n/a"


def _signed(value: str, digits: int = 2) -> str:
    return f"{float(value):+.{digits}f}" if value != "" else "n/a"


LIMITATIONS = [
    "Words are whitespace tokens, so punctuation attached to a word is not counted separately and the JSON keys of "
    "the 21 malformed GPT captions count as words; the valid-for-all sample excludes those recordings.",
    "Model captions are scored against five references (E12) and held-out human captions against four (E13), so "
    "model and human correlations come from different reference sets.",
    "Correlations are associations over recordings, not effects of length: length varies with the recording and the "
    "content of the caption. Formal tests and corrections for multiple comparisons belong to E16.",
    "CIs are 95% percentile intervals per comparison, uncorrected.",
]


def report_markdown(report: dict[str, Any]) -> str:
    """The caption-length report: numbers, IDs and settings only, never caption text."""
    names = {**report["display_names"], HUMAN: "Human"}
    systems = list(names)
    models = systems[:-1]
    run_info, inputs, sample = report["run_info"], report["inputs"], report["samples"]
    by = {(r["system"], r["sample"]): r for r in report["summary"]}
    all_rows = [by[(s, sample["all"])] for s in systems]
    lines = [
        "# Caption-length analysis (E15)",
        "",
        f"Generated by `scripts/analyze_lengths.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}), {run_info['timestamp_utc']}.",
        "",
        "## Inputs",
        "",
        f"- {inputs['recordings']} recordings; {len(models)} models with one caption each and "
        f"{inputs['references_per_recording']} human references per recording.",
        f"- Captions: `{inputs['captions']['path']}` (sha256 `{inputs['captions']['sha256']}`), the scored `caption` "
        f"field; references: `{inputs['references']['path']}` (sha256 `{inputs['references']['sha256']}`).",
        f"- Scores: `{inputs['model_scores']['path']}` (sha256 `{inputs['model_scores']['sha256']}`) and "
        f"`{inputs['human_scores']['path']}` (sha256 `{inputs['human_scores']['sha256']}`).",
        f"- Words: {report['word_definition']}",
        f"- Bootstrap: {report['bootstrap']['num_resamples']} resamples of recordings, "
        f"{report['bootstrap']['confidence']:.0%} percentile intervals, seed {report['bootstrap']['seed']}.",
        "",
        "## Caption length (words)",
        "",
        "| System | Captions | Mean | SD | Min | Q1 | Median | Q3 | IQR | Max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_rows:
        stats = " | ".join(_fmt(r[k]) for k in ("mean", "sd", "min", "q1", "median", "q3", "iqr", "max"))
        lines.append(f"| {names[r['system']]} | {r['num_captions']} | {stats} |")
    lines += [
        "",
        "## Model-human comparison",
        "",
        "Ratio of mean lengths (model / human references), and the mean per-recording difference between the model "
        "caption and the mean of its recording's five references, with bootstrap CIs. The shares compare each caption "
        "with the references of its recording (for human captions, the other four).",
        "",
        "| System | Ratio to human [95% CI] | Words minus reference mean [95% CI] | Longer than reference mean | "
        "Longer than every reference |",
        "|---|---|---|---|---|",
    ]
    for r in all_rows:
        ratio = "n/a" if r["ratio_to_human"] == "" else f"{_fmt(r['ratio_to_human'])} "
        if r["ratio_to_human"] != "":
            ratio += f"[{_fmt(r['ratio_ci_low'])}, {_fmt(r['ratio_ci_high'])}]"
        diff = "n/a" if r["difference_mean"] == "" else f"{_signed(r['difference_mean'])} "
        if r["difference_mean"] != "":
            diff += f"[{_signed(r['difference_ci_low'])}, {_signed(r['difference_ci_high'])}]"
        longer = (float(r["share_longer_than_reference_mean"]), float(r["share_longer_than_all_references"]))
        shares = f"{longer[0]:.1%} | {longer[1]:.1%}"
        lines.append(f"| {names[r['system']]} | {ratio} | {diff} | {shares} |")
    valid = [by[(s, sample["valid"])] for s in systems if (s, sample["valid"]) in by]
    lines += [
        "",
        f"### Sensitivity: {valid[0]['num_recordings']} recordings where every model caption is valid",
        "",
        "| System | Mean words (all) | Mean words (valid) | Ratio to human (all) | Ratio to human (valid) |",
        "|---|---|---|---|---|",
    ]
    for r in valid:
        full = by[(r["system"], sample["all"])]
        cells = [_fmt(full["mean"]), _fmt(r["mean"]), _fmt(full["ratio_to_human"]), _fmt(r["ratio_to_human"])]
        lines.append(f"| {names[r['system']]} | " + " | ".join(cells) + " |")
    assoc = {(r["system"], r["predictor"], r["metric"]): r for r in report["associations"] if r["sample"] == "all"}
    metrics = report["association_metrics"]
    for predictor, title in (
        ("words", "caption length"),
        ("words_minus_reference_mean", "length minus the mean length of the references"),
    ):
        lines += [
            "",
            f"## Spearman correlation between {title} and score",
            "",
            "Per caption, over all recordings; models scored against five references (E12), held-out human captions "
            "against four (E13, all five positions pooled). `*` marks a 95% CI that excludes 0 (uncorrected).",
            "",
            "| Metric | " + " | ".join(names[s] for s in systems) + " |",
            "|---|" + "---|" * len(systems),
        ]
        for metric in metrics:
            cells = []
            for system in systems:
                r = assoc[(system, predictor, metric)]
                mark = "*" if r["excludes_zero"] else ""
                cells.append(f"{_signed(r['spearman'])}{mark} [{_signed(r['ci_low'])}, {_signed(r['ci_high'])}]")
            lines.append(f"| {BY_NAME[metric].label} | " + " | ".join(cells) + " |")
    lines += ["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""]
    lines += ["## Outputs", "", *[f"- `{path}`" for path in report["outputs"].values()], ""]
    return "\n".join(lines)
