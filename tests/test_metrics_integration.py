"""Metric integration tests (E11): the real jars, JDK and checkpoints.

Run with `make test-metrics` (.venv-metrics and `make metric-tools`). Skipped elsewhere, including CI. The golden values
were measured in the E11 metric check with aac-metrics 0.5.5 on Temurin 11.0.32.1 and reproduced by this pipeline.
"""

import importlib.util
import os

import numpy as np
import pytest

from src.metrics.tools import activate, tool_paths, verify_manifest
from src.utils.config import load_config, load_env

pytestmark = pytest.mark.metrics
CONFIG = load_config("metrics")
if importlib.util.find_spec("aac_metrics") is None:
    pytest.skip("needs .venv-metrics (make test-metrics)", allow_module_level=True)
if verify_manifest(tool_paths(CONFIG["tools"])["root"]):
    pytest.skip("metric tools missing or changed (make metric-tools)", allow_module_level=True)
# Before anything imports huggingface_hub: models must load from tools/ at the pinned revisions, never be downloaded.
activate(CONFIG["tools"], offline=True)

CANDIDATES = {
    "a": "a dog barks loudly while cars pass by on a busy road",
    "b": "rain falls steadily on a metal roof",
    "c": "a man speaks while birds chirp in the background",
}
REFERENCES = {
    "a": [
        "a dog is barking near a busy street",
        "cars drive by as a dog barks",
        "a dog barks repeatedly as traffic passes",
        "traffic noise with a dog barking",
        "a barking dog and passing vehicles",
    ],
    "b": [
        "heavy rain is falling on a roof",
        "rain hits a tin roof",
        "rain pours down onto a metal surface",
        "steady rainfall on a rooftop",
        "raindrops fall on a metal roof continuously",
    ],
    "c": [
        "a person talks as birds sing",
        "birds chirp while a man is speaking",
        "a man talks outdoors with birds chirping",
        "someone speaks and birds tweet",
        "a male voice speaks while birds are singing",
    ],
}
GOLDEN_CORPUS = {
    "bleu_1": 0.6429,
    "bleu_2": 0.5071,
    "bleu_3": 0.3602,
    "bleu_4": 0.2227,
    "meteor": 0.3300,
    "cider_d": 1.3336,
    "spice": 0.3326,
    "spider": 0.8331,
    "spider_fl": 0.4972,
    "sbert_similarity": 0.8325,
    "fluency_error": 0.3333,
    "fense": 0.5792,
    "bertscore_f1": 0.9849,  # 0.9817 before the E21 fix (torchmetrics paired captions with other references)
}
GOLDEN_SPICE_ITEMS = [0.2609, 0.4211, 0.3158]
IDS = list(CANDIDATES)


@pytest.fixture(scope="module")
def evaluator():
    from src.evaluation.scoring import Evaluator

    return Evaluator(CONFIG["settings"], CONFIG["tools"])


@pytest.fixture(scope="module")
def golden(evaluator):
    return evaluator.score("test", "five_reference", "all", IDS, CANDIDATES, REFERENCES)


def test_java_is_the_pinned_jdk(evaluator):
    from src.metrics.tools import check_java

    assert check_java(evaluator.java).startswith("11.")


def test_golden_values(golden):
    corpus = {name: round(score.corpus, 4) for name, score in golden.scores.items() if name in GOLDEN_CORPUS}
    assert corpus == GOLDEN_CORPUS
    assert golden.scores["spice"].items.round(4).tolist() == GOLDEN_SPICE_ITEMS


def test_corpus_equals_item_mean_where_registered(golden):
    from src.metrics.registry import BY_NAME

    for name, score in golden.scores.items():
        if BY_NAME[name].corpus_is_item_mean:
            assert score.corpus == pytest.approx(score.items.mean(), abs=1e-12), name


def test_bertscore_rescaling_uses_the_layer_24_baseline(golden):
    from src.metrics.reference_based import read_bertscore_baseline

    baseline = read_bertscore_baseline(tool_paths(CONFIG["tools"])["bertscore_baseline"], 24)
    raw, rescaled = golden.scores["bertscore_f1"].items, golden.scores["bertscore_f1_rescaled"].items
    assert rescaled == pytest.approx((raw - baseline) / (1 - baseline))
    assert baseline == pytest.approx(0.960168)  # layer 24 (layer 17, bert_score's default, is 0.8312)


def test_sanity_pairs(evaluator):
    candidates = {"same": "a dog barks at a passing car", "disjoint": "violin music plays softly"}
    references = {"same": [candidates["same"]] * 5, "disjoint": ["a dog is barking loudly"] * 5}
    scores = evaluator.score("test", "five_reference", "all", list(candidates), candidates, references).scores
    for n in range(1, 5):
        assert scores[f"bleu_{n}"].items[0] == pytest.approx(1.0)
    assert scores["bleu_1"].items[1] == pytest.approx(0.0, abs=1e-12)  # aac-metrics BLEU smooths with 1e-15
    assert scores["spice"].items.tolist() == [1.0, 0.0]
    assert scores["bertscore_f1"].items[0] == pytest.approx(1.0, abs=1e-5)


def test_json_like_caption_is_scored_as_returned(evaluator):
    from src.metrics.preprocess import ptb_tokenize

    caption = '{"main_sound_sources": "dog", "audible_events": "barking and cars passing"}'
    paths = tool_paths(CONFIG["tools"])
    root = paths["root"]
    (tokenized,) = ptb_tokenize([caption], java=evaluator.java, cache_path=root, tmp_path=root / "tmp")
    assert not set('{}":') & set(tokenized)
    assert "main_sound_sources" in tokenized  # keys are kept: that is the penalty for the format
    candidates = {"a": caption, "b": CANDIDATES["b"]}
    references = {"a": REFERENCES["a"], "b": REFERENCES["b"]}
    scores = evaluator.score("test", "five_reference", "all", ["a", "b"], candidates, references).scores
    assert all(np.isfinite(score.items).all() for score in scores.values())


def test_identical_scores_on_rerun(golden):
    from src.evaluation.scoring import Evaluator

    again = Evaluator(CONFIG["settings"], CONFIG["tools"]).score(
        "test", "five_reference", "all", IDS, CANDIDATES, REFERENCES
    )
    for name, score in golden.scores.items():
        assert np.array_equal(score.items, again.scores[name].items), name
        assert score.corpus == again.scores[name].corpus, name


def test_four_reference_subset_changes_only_reference_dependent_scores(evaluator, golden):
    from src.evaluation.references import subset_references
    from src.metrics.registry import REFERENCE_FREE

    four = evaluator.score("test", "four_reference", "drop_1", IDS, CANDIDATES, subset_references(REFERENCES, "drop_1"))
    for name in ("fluency_error",):
        assert np.array_equal(four.scores[name].items, golden.scores[name].items)
    assert REFERENCE_FREE >= {"fluency_error"}
    assert not np.array_equal(four.scores["cider_d"].items, golden.scores["cider_d"].items)


@pytest.fixture(scope="module")
def pilot_audio():
    from src.data.clotho import read_jsonl
    from src.evaluation.inputs import audio_paths

    load_env()
    if not os.environ.get("CLOTHO_ROOT"):
        pytest.skip("CLOTHO_ROOT is not set")
    manifest = read_jsonl("data/manifests/clotho_pilot_manifest.jsonl")[:3]
    return manifest, audio_paths(manifest, os.environ["CLOTHO_ROOT"])


def test_audio_aware_scores_match_aac_metrics_text_mace(evaluator, pilot_audio):
    from aac_metrics.functional import mace

    manifest, audio = pilot_audio
    ids = [record["sample_id"] for record in manifest]
    candidates = {record["sample_id"]: record["captions"][0] for record in manifest}
    references = {record["sample_id"]: record["captions"][1:] for record in manifest}
    scores = evaluator.score("human", "human_loo", "drop_1", ids, candidates, references, audio).scores
    for name in ("clap_similarity", "mace", "mace_text", "mace_audio"):
        assert len(set(scores[name].items.tolist())) == len(ids), name  # per-item values, not the corpus mean
        assert np.all((scores[name].items > -1) & (scores[name].items <= 1)), name

    # Our text-only MACE equals aac-metrics' MACE (text mode) on the same tokenized inputs and CLAP model.
    from src.metrics.preprocess import ptb_tokenize

    paths = tool_paths(CONFIG["tools"])
    tokenize = lambda texts: ptb_tokenize(  # noqa: E731
        texts, java=evaluator.java, cache_path=paths["root"], tmp_path=paths["root"] / "tmp"
    )
    cands = tokenize([candidates[sid] for sid in ids])
    refs = [tokenize(references[sid]) for sid in ids]
    corpus, _ = mace(cands, refs, mace_method="text", clap_model=evaluator.clap.model, device="cpu")
    assert scores["mace_text"].corpus == pytest.approx(float(corpus["mace"]), abs=1e-5)


def test_clap_audio_embedding_covers_the_clip(evaluator, pilot_audio):
    from src.metrics.clap import chunk_audio, read_mono

    manifest, audio = pilot_audio
    record = manifest[0]
    samples = read_mono(audio[record["sample_id"]], 44100)
    assert samples.size == record["num_frames"]
    chunks, lengths = chunk_audio(samples, evaluator.clap.chunk_size)
    assert lengths.sum() == record["num_frames"] and len(chunks) == -(-record["num_frames"] // (7 * 44100))
    first = evaluator.clap.audio([audio[record["sample_id"]]])
    evaluator.clap._audio.clear()
    assert np.array_equal(first, evaluator.clap.audio([audio[record["sample_id"]]]))


def test_bertscore_does_not_depend_on_the_other_pairs_in_the_call(evaluator):
    """Regression (E21): torchmetrics 1.9.0 reorders embeddings with the sort permutation instead of its inverse, so a
    batched call scored candidates against other pairs' references. Each pair must score as it does alone."""
    from aac_metrics.functional.bert_score_mrefs import bert_score_mrefs

    from src.metrics.bertscore import bertscore_max, bertscore_pairs

    model, tokenizer = evaluator.reference._model("bert")
    kw = {"model": model, "tokenizer": tokenizer, "device": "cpu", "max_length": 128}
    candidates = [CANDIDATES[i] for i in IDS] + ["birds"]
    references = [REFERENCES[i] for i in IDS] + [["many small birds are singing loudly in a large garden at dawn"]]
    together = bertscore_max(candidates, references, batch_size=2, **kw)
    alone = [bertscore_max([c, c], [r, r], batch_size=32, **kw)[0] for c, r in zip(candidates, references, strict=True)]
    assert together == pytest.approx(alone, abs=1e-6)
    # torchmetrics on one pair (duplicated, so nothing can be misordered) agrees with ours
    pairs = [(c, r) for c, refs in zip(candidates, references, strict=True) for r in refs]
    for c, r in pairs[:3] + pairs[-1:]:
        expected = float(bert_score_mrefs([c, c], [[r], [r]], batch_size=32, idf=False, reduction="max", **kw)[1]
                         ["bert_score.f1"][0])  # fmt: skip
        assert bertscore_pairs([c], [r], batch_size=32, **kw)[0] == pytest.approx(expected, abs=1e-6)
