"""Reference-based caption metrics (E11): BLEU-1..4, METEOR, CIDEr-D, SPICE, SPIDEr, SPIDEr-FL, FENSE, BERTScore.

Inputs are PTB-tokenized (`src.metrics.preprocess`). BLEU, METEOR, CIDEr-D, SBERT similarity, the fluency-error detector
and BERTScore come from aac-metrics 0.5.5; SPICE runs through `src.metrics.spice`; SPIDEr, SPIDEr-FL, FENSE and the
rescaled BERTScore are combined here from their parts with aac-metrics' formulas. One call scores a whole corpus:
CIDEr-D document frequencies and the BLEU and METEOR corpus statistics depend on every reference in the call.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics.registry import Score, Timer, check_scores
from src.metrics.spice import spice_scores
from src.metrics.tools import java_path, tool_paths


def flag_errors(error_probs: np.ndarray, threshold: float) -> np.ndarray:
    """1.0 where the fluency-error probability exceeds `threshold` (strictly, as in aac-metrics), else 0.0."""
    return (np.asarray(error_probs) > threshold).astype(np.float64)


def penalize(scores: np.ndarray, errors: np.ndarray, penalty: float) -> np.ndarray:
    """FENSE-style penalty: scores * (1 - penalty * error) per item."""
    return np.asarray(scores, dtype=np.float64) * (1.0 - penalty * np.asarray(errors, dtype=np.float64))


def read_bertscore_baseline(path: str | Path, layer: int) -> float:
    """F baseline for `layer` from a bert_score rescale-baseline file (columns LAYER, P, R, F)."""
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["LAYER"]) == layer:
                return float(row["F"])
    raise ValueError(f"No layer {layer} in {path}")


def rescale(values: np.ndarray, baseline: float) -> np.ndarray:
    """BERTScore baseline rescaling: (x - b) / (1 - b)."""
    return (np.asarray(values, dtype=np.float64) - baseline) / (1.0 - baseline)


def _numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _mean_score(items: np.ndarray) -> Score:
    return Score(float(items.mean()), items)


class ReferenceScorer:
    """Scores PTB-tokenized candidates against reference lists; models are loaded once and reused across calls."""

    def __init__(self, settings: dict[str, Any], tools: dict[str, Any], timer: Timer | None = None) -> None:
        self.settings = settings
        self.timer = timer or Timer()
        self.paths = tool_paths(tools)
        self.java = str(java_path(tools))
        self.device = settings["device"]
        self.batch_size = settings["batch_size"]
        bertscore = settings["bertscore"]
        self.bertscore_baseline = read_bertscore_baseline(self.paths["bertscore_baseline"], bertscore["baseline_layer"])
        self._models: dict[str, Any] = {}
        self._error_probs: dict[str, float] = {}

    def load_models(self) -> None:
        for name in ("echecker", "sbert", "bert"):
            self._model(name)

    def _model(self, name: str) -> Any:
        if name in self._models:
            return self._models[name]
        with self.timer("load_models"):
            if name == "echecker":
                from aac_metrics.functional.fer import _load_echecker_and_tokenizer

                self._models[name] = _load_echecker_and_tokenizer(
                    echecker=self.settings["fluency_error"]["model"],
                    echecker_tokenizer=None,
                    device=self.device,
                    reset_state=True,
                    verbose=0,
                )
            elif name == "sbert":
                from sentence_transformers import SentenceTransformer

                self._models[name] = SentenceTransformer(self.settings["fense"]["sbert_model"], device=self.device)
            elif name == "bert":
                from aac_metrics.functional.bert_score_mrefs import _load_model_and_tokenizer

                self._models[name] = _load_model_and_tokenizer(
                    model=self.settings["bertscore"]["model"],
                    tokenizer=None,
                    device=self.device,
                    reset_state=True,
                    verbose=0,
                )
            else:
                raise KeyError(name)
        return self._models[name]

    def error_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Fluency-error probability per text (the detector's `error` output); each distinct text is scored once."""
        from aac_metrics.functional import fer

        echecker, tokenizer = self._model("echecker")
        if new := sorted(set(texts) - set(self._error_probs)):
            with self.timer("fluency_error"):
                _, sents = fer(
                    new,
                    echecker=echecker,
                    echecker_tokenizer=tokenizer,
                    device=self.device,
                    batch_size=self.batch_size,
                    return_probs=True,
                )
            self._error_probs.update(zip(new, _numpy(sents["fluency_error.error_prob"]).tolist(), strict=True))
        return np.array([self._error_probs[text] for text in texts], dtype=np.float64)

    def score(self, candidates: Sequence[str], mult_references: Sequence[Sequence[str]]) -> dict[str, Score]:
        from aac_metrics.functional import bert_score_mrefs, bleu, cider_d, meteor, sbert_sim

        candidates, mult_references = list(candidates), [list(refs) for refs in mult_references]
        settings, scores, timer = self.settings, {}, self.timer

        with timer("bleu"):
            option = settings["bleu"]["option"]
            corpus, sents = bleu(candidates, mult_references, n=4, option=option, return_1_to_n=True)
        bleu_corpus, bleu_items = _numpy(corpus["bleu_1_to_4"]), _numpy(sents["bleu_1_to_4"])
        for n in range(1, 5):
            scores[f"bleu_{n}"] = Score(float(bleu_corpus[n - 1]), bleu_items[n - 1])
        with timer("meteor"):
            corpus, sents = meteor(
                candidates,
                mult_references,
                cache_path=str(self.paths["root"]),
                java_path=self.java,
                java_max_memory=settings["meteor"]["java_max_memory"],
            )
        scores["meteor"] = Score(float(_numpy(corpus["meteor"])), _numpy(sents["meteor"]))
        cider = settings["cider_d"]
        with timer("cider_d"):
            _, sents = cider_d(candidates, mult_references, n=cider["n"], sigma=cider["sigma"], scale=cider["scale"])
        scores["cider_d"] = _mean_score(_numpy(sents["cider_d"]))
        with timer("spice"):
            spice = spice_scores(
                candidates,
                mult_references,
                java=self.java,
                tools_dir=self.paths["root"],
                java_max_memory=settings["spice"]["java_max_memory"],
                threads=settings["spice"]["threads"],
            )
        scores["spice"] = _mean_score(spice)
        spider = (scores["cider_d"].items + spice) / 2.0
        scores["spider"] = _mean_score(spider)

        error_probs = self.error_probabilities(candidates)
        fl = settings["spider_fl"]
        spider_fl_errors = flag_errors(error_probs, fl["error_threshold"])
        scores["spider_fl"] = _mean_score(penalize(spider, spider_fl_errors, fl["penalty"]))
        sbert_model = self._model("sbert")
        with timer("sbert"):
            _, sents = sbert_sim(
                candidates, mult_references, sbert_model=sbert_model, device=self.device, batch_size=self.batch_size
            )
        sbert = _numpy(sents["sbert_sim"])
        fense = settings["fense"]
        fense_errors = flag_errors(error_probs, fense["error_threshold"])
        scores["sbert_similarity"] = _mean_score(sbert)
        scores["fluency_error"] = _mean_score(fense_errors)
        scores["fense"] = _mean_score(penalize(sbert, fense_errors, fense["penalty"]))

        model, tokenizer = self._model("bert")
        bertscore = settings["bertscore"]
        with timer("bertscore"):
            _, sents = bert_score_mrefs(
                candidates,
                mult_references,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                batch_size=self.batch_size,
                max_length=bertscore["max_length"],
                idf=bertscore["idf"],
                reduction=bertscore["reduction"],
            )
        f1 = _numpy(sents["bert_score.f1"])
        scores["bertscore_f1"] = _mean_score(f1)
        scores["bertscore_f1_rescaled"] = _mean_score(rescale(f1, self.bertscore_baseline))

        if errors := check_scores(scores, len(candidates)):
            raise RuntimeError(f"Invalid reference-based scores: {errors}")
        return scores
