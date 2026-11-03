"""Score one system under one reference subset, and write long-format score tables (E11-E14).

A scoring call covers every recording of the condition at once (see `src.metrics.reference_based`). Candidates and
references are PTB-tokenized together in one Java call; audio-aware metrics are added when audio paths are given.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.inputs import check_condition_inputs
from src.metrics.registry import METRICS, Score

SCORE_COLUMNS = ("system", "condition", "subset", "sample_id", "metric", "score")
CORPUS_COLUMNS = ("system", "condition", "subset", "metric", "corpus", "item_mean", "num_items")


@dataclass
class ScoreSet:
    system: str
    condition: str
    subset: str
    sample_ids: list[str]
    scores: dict[str, Score]


class Evaluator:
    """Scores candidates with every metric; models are loaded once and reused across systems and subsets."""

    def __init__(self, settings: dict[str, Any], tools: dict[str, Any]) -> None:
        from src.metrics.reference_based import ReferenceScorer
        from src.metrics.tools import activate, java_path, tool_paths

        activate(tools, offline=True)
        self.settings, self.tools = settings, tools
        self.paths = tool_paths(tools)
        self.java = java_path(tools)
        (self.paths["root"] / "tmp").mkdir(parents=True, exist_ok=True)
        self.reference = ReferenceScorer(settings, tools)
        self._clap = None

    @property
    def clap(self):
        if self._clap is None:
            from src.metrics.clap import ClapEmbedder
            from src.metrics.tools import clap_weights_path

            self._clap = ClapEmbedder(clap_weights_path(self.tools), self.settings["clap"], self.settings["batch_size"])
        return self._clap

    def score(
        self,
        system: str,
        condition: str,
        subset: str,
        sample_ids: list[str],
        candidates: dict[str, str],
        references: dict[str, list[str]],
        audio: dict[str, Path] | None = None,
    ) -> ScoreSet:
        from src.metrics.mace import audio_aware_scores
        from src.metrics.preprocess import ptb_tokenize

        num_references = len(references[sample_ids[0]]) if sample_ids else 0
        if errors := check_condition_inputs(sample_ids, candidates, references, num_references):
            raise ValueError(f"{system} {condition} {subset}: invalid inputs: {errors[:5]}")
        raw_candidates = [candidates[sid] for sid in sample_ids]
        raw_references = [references[sid] for sid in sample_ids]
        flat = raw_candidates + [ref for refs in raw_references for ref in refs]
        root = self.paths["root"]
        tokenized = ptb_tokenize(flat, java=self.java, cache_path=root, tmp_path=root / "tmp")
        cands = tokenized[: len(sample_ids)]
        refs = [
            tokenized[len(sample_ids) + i * num_references : len(sample_ids) + (i + 1) * num_references]
            for i in range(len(sample_ids))
        ]

        scores = self.reference.score(cands, refs)
        if audio is not None:
            clap = self.clap
            clap.text(cands + [ref for item in refs for ref in item])  # embed all distinct texts in batches
            scores |= audio_aware_scores(
                clap.text(cands),
                [clap.text(item) for item in refs],
                clap.audio([audio[sid] for sid in sample_ids]),
                self.reference.error_probabilities(cands),
                self.settings["mace"],
            )
        ordered = {metric.name: scores[metric.name] for metric in METRICS if metric.name in scores}
        return ScoreSet(system, condition, subset, list(sample_ids), ordered)


def score_rows(score_set: ScoreSet) -> list[dict[str, Any]]:
    rows = []
    for metric, score in score_set.scores.items():
        for sample_id, value in zip(score_set.sample_ids, score.items.tolist(), strict=True):
            rows.append(
                {
                    "system": score_set.system,
                    "condition": score_set.condition,
                    "subset": score_set.subset,
                    "sample_id": sample_id,
                    "metric": metric,
                    "score": repr(float(value)),
                }
            )
    return rows


def corpus_rows(score_set: ScoreSet) -> list[dict[str, Any]]:
    return [
        {
            "system": score_set.system,
            "condition": score_set.condition,
            "subset": score_set.subset,
            "metric": metric,
            "corpus": repr(float(score.corpus)),
            "item_mean": repr(float(score.items.mean())),
            "num_items": len(score.items),
        }
        for metric, score in score_set.scores.items()
    ]


def write_csv(rows: list[dict[str, Any]], path: str | Path, columns: tuple[str, ...]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path
