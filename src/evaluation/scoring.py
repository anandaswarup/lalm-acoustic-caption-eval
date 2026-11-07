"""Score one system under one reference subset, and write and read long-format score tables (E10-E13).

A scoring call covers every recording of the condition at once (see `src.metrics.reference_based`). Candidates and
references are PTB-tokenized together in one Java call; audio-aware metrics are added when audio paths are given.
Tables ending in `.gz` are gzip-compressed, with no timestamp in the header so that equal tables give equal files.
"""

from __future__ import annotations

import csv
import gzip
import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from src.evaluation.inputs import check_condition_inputs
from src.metrics.registry import METRICS, Score, Timer

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
        self.timer = Timer()
        self.reference = ReferenceScorer(settings, tools, self.timer)
        self._clap = None

    def load_models(self, audio: bool = True) -> None:
        """Load every model up front, so that per-metric timings exclude loading."""
        self.reference.load_models()
        if audio:
            _ = self.clap

    @property
    def clap(self):
        if self._clap is None:
            from src.metrics.clap import ClapEmbedder
            from src.metrics.tools import clap_weights_path

            with self.timer("load_models"):
                weights, settings = clap_weights_path(self.tools), self.settings["clap"]
                self._clap = ClapEmbedder(weights, settings, self.settings["batch_size"])
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
        with self.timer("ptb_tokenize"):
            tokenized = ptb_tokenize(flat, java=self.java, cache_path=root, tmp_path=root / "tmp")
        cands = tokenized[: len(sample_ids)]
        refs = [
            tokenized[len(sample_ids) + i * num_references : len(sample_ids) + (i + 1) * num_references]
            for i in range(len(sample_ids))
        ]

        scores = self.reference.score(cands, refs)
        if audio is not None:
            clap = self.clap
            with self.timer("clap_text"):
                clap.text(cands + [ref for item in refs for ref in item])  # embed all distinct texts in batches
            with self.timer("clap_audio"):
                audio_embeddings = clap.audio([audio[sid] for sid in sample_ids])
            scores |= audio_aware_scores(
                clap.text(cands),
                [clap.text(item) for item in refs],
                audio_embeddings,
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


@contextmanager
def _open_text(path: Path, mode: str) -> Iterator[TextIO]:
    """A CSV text stream, gzip-compressed when `path` ends in `.gz` (mode "r" or "w")."""
    if path.suffix != ".gz":
        with path.open(mode, encoding="utf-8", newline="") as stream:
            yield stream
        return
    with path.open(mode + "b") as raw, gzip.GzipFile(filename="", mode=mode + "b", fileobj=raw, mtime=0) as packed:
        with io.TextIOWrapper(packed, encoding="utf-8", newline="") as stream:
            yield stream


def write_csv(rows: list[dict[str, Any]], path: str | Path, columns: tuple[str, ...]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with _open_text(Path(path), "r") as stream:
        return list(csv.DictReader(stream))


def read_score_sets(scores_path: str | Path, corpus_path: str | Path) -> list[ScoreSet]:
    """Score sets back from a per-item table and its corpus table (the inverse of `score_rows` and `corpus_rows`).

    One set per (system, condition, subset) of the per-item table, in table order; the corpus table may hold more
    (e.g. averaged subsets), which are ignored. Values round-trip exactly (they are written with `repr`).
    """
    corpus = {
        (r["system"], r["condition"], r["subset"], r["metric"]): float(r["corpus"]) for r in read_csv(corpus_path)
    }
    groups: dict[tuple[str, str, str], dict[str, dict[str, float]]] = {}
    for row in read_csv(scores_path):
        metric = groups.setdefault((row["system"], row["condition"], row["subset"]), {}).setdefault(row["metric"], {})
        if row["sample_id"] in metric:
            raise ValueError(f"{scores_path}: duplicate score {row['system']} {row['subset']} {row['sample_id']}")
        metric[row["sample_id"]] = float(row["score"])
    order = {metric.name: index for index, metric in enumerate(METRICS)}
    score_sets = []
    for (system, condition, subset), by_metric in groups.items():
        if unknown := sorted(set(by_metric) - set(order)):
            raise ValueError(f"{scores_path}: unknown metrics {unknown}")
        names = sorted(by_metric, key=order.__getitem__)
        sample_ids = list(by_metric[names[0]])
        scores = {}
        for name in names:
            if list(by_metric[name]) != sample_ids:
                raise ValueError(f"{scores_path}: {system} {subset} {name} scores different recordings")
            if (key := (system, condition, subset, name)) not in corpus:
                raise ValueError(f"{corpus_path}: no corpus value for {' '.join(key)}")
            scores[name] = Score(corpus[key], np.array(list(by_metric[name].values()), dtype=np.float64))
        score_sets.append(ScoreSet(system, condition, subset, sample_ids, scores))
    return score_sets
