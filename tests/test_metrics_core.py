"""Metric arithmetic and parsing (E11) that needs no Java, models or downloads."""

import math

import numpy as np
import pytest

from src.metrics.clap import chunk_audio, cosine_rows, mean_reference_cosine, weighted_mean
from src.metrics.mace import mace_scores
from src.metrics.preprocess import check_texts
from src.metrics.reference_based import flag_errors, penalize, read_bertscore_baseline, rescale
from src.metrics.registry import BY_NAME, METRICS, Score, check_scores
from src.metrics.spice import parse_spice_output, spice_input
from src.utils.config import load_config


def spice_item(index, f):
    return {"image_id": index, "scores": {"All": {"f": f, "pr": 0.5, "re": 0.5}}}


def test_spice_input_format():
    items = spice_input(["a dog barks"], [["a dog is barking", "dog barks"]])
    assert items == [{"image_id": 0, "test": "a dog barks", "refs": ["a dog is barking", "dog barks"]}]
    with pytest.raises(ValueError):
        spice_input(["a", "b"], [["x"]])


def test_parse_spice_output_orders_by_id():
    scores = parse_spice_output([spice_item(1, 0.25), spice_item(0, "0.5")], 2)
    assert scores.tolist() == [0.5, 0.25] and scores.dtype == np.float64


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([spice_item(0, 0.5)], "no score for 1 items"),
        ([spice_item(0, 0.5), spice_item(0, 0.5)], "twice"),
        ([spice_item(0, 0.5), spice_item(1, "NaN")], "scored nan"),
        ([spice_item(0, 0.5), spice_item(1, None)], "no numeric"),
        ([spice_item(0, 0.5), spice_item(1, 0.5), spice_item(2, 0.5)], "unknown items"),
    ],
)
def test_parse_spice_output_rejects_bad_results(results, message):
    with pytest.raises(RuntimeError, match=message):
        parse_spice_output(results, 2)


def test_fluency_penalty_is_strict_and_multiplicative():
    errors = flag_errors(np.array([0.9, 0.9000001, 0.1]), 0.9)
    assert errors.tolist() == [0.0, 1.0, 0.0]
    assert penalize(np.array([0.8, 0.8, 0.8]), errors, 0.9).tolist() == pytest.approx([0.8, 0.08, 0.8])


def test_bertscore_rescaling(tmp_path):
    baseline = tmp_path / "roberta-large.tsv"
    baseline.write_text("LAYER,P,R,F\n0,0.37,0.37,0.36\n24,0.83,0.83,0.8\n", encoding="utf-8")
    b = read_bertscore_baseline(baseline, 24)
    assert b == 0.8
    assert rescale(np.array([0.8, 0.9, 1.0]), b).tolist() == pytest.approx([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="No layer 17"):
        read_bertscore_baseline(baseline, 17)


def test_pinned_baseline_layer_matches_aac_metrics_bertscore():
    # aac-metrics passes roberta-large to torchmetrics without num_layers, which then uses the last of its 24 layers.
    assert load_config("metrics")["settings"]["bertscore"]["baseline_layer"] == 24


def test_chunk_audio_covers_the_whole_clip():
    samples = np.arange(10, dtype=np.float32)
    chunks, lengths = chunk_audio(samples, 4)
    assert chunks.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 8, 9]]
    assert lengths.tolist() == [4, 4, 2]
    chunks, lengths = chunk_audio(samples[:8], 4)
    assert chunks.shape == (2, 4) and lengths.tolist() == [4, 4]


def test_chunk_audio_repeats_a_short_clip():
    chunks, lengths = chunk_audio(np.array([1.0, 2.0, 3.0]), 7)
    assert chunks.tolist() == [[1, 2, 3, 1, 2, 3, 1]] and lengths.tolist() == [3]
    with pytest.raises(ValueError):
        chunk_audio(np.zeros((2, 3)), 4)
    with pytest.raises(ValueError):
        chunk_audio(np.array([]), 4)


def test_weighted_mean_by_duration():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert weighted_mean(embeddings, np.array([3.0, 1.0])).tolist() == [0.75, 0.25]


def test_cosines():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [-1.0, -1.0]])
    assert cosine_rows(a, b).tolist() == pytest.approx([1.0, -1.0])
    refs = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 1.0]])]
    assert mean_reference_cosine(a, refs).tolist() == pytest.approx([0.5, 1.0])


def test_mace_follows_the_paper_per_item():
    audio_text = np.array([0.4, 0.6, 0.2])
    text_text = np.array([0.6, 0.2, 0.4])
    probs = np.array([0.1, 0.98, 0.97])  # only the second exceeds 0.97
    scores = mace_scores(audio_text, text_text, probs, error_threshold=0.97, penalty=0.3)
    assert scores["mace"].items.tolist() == pytest.approx([0.5, 0.4 * 0.7, 0.3])
    assert scores["mace_text"].items.tolist() == pytest.approx([0.6, 0.2 * 0.7, 0.4])
    assert scores["mace_audio"].items.tolist() == pytest.approx([0.4, 0.6 * 0.7, 0.2])
    assert scores["clap_similarity"].items.tolist() == audio_text.tolist()
    assert scores["fluency_error_mace"].items.tolist() == [0.0, 1.0, 0.0]
    # Regression for aac-metrics 0.5.5, whose per-item MACE holds the corpus mean.
    assert len(set(scores["mace"].items.tolist())) == 3
    assert scores["mace"].corpus == pytest.approx(scores["mace"].items.mean())


def test_check_texts():
    assert check_texts(["a dog barks", '{"a": "b"}'], "c") == []
    assert check_texts(["", " ", "a\nb", None], "c") == [
        "c[0]: empty",
        "c[1]: empty",
        "c[2]: contains a line break",
        "c[3]: not a string (NoneType)",
    ]


def test_check_scores():
    good = Score(0.5, np.array([0.4, 0.6]))
    assert check_scores({"m": good}, 2) == []
    assert "item scores for 3 items" in check_scores({"m": good}, 3)[0]
    assert "non-finite item" in check_scores({"m": Score(0.5, np.array([0.4, math.nan]))}, 2)[0]
    assert "non-finite corpus" in check_scores({"m": Score(math.inf, np.array([0.4, 0.6]))}, 2)[0]


def test_registry():
    names = [metric.name for metric in METRICS]
    assert len(names) == len(set(names)) == len(BY_NAME)
    assert {name for name in names if not BY_NAME[name].corpus_is_item_mean} == {
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
    }
    primary = {metric.name for metric in METRICS if metric.family != "auxiliary"}
    assert {"spider_fl", "fense", "bertscore_f1", "clap_similarity", "mace"} <= primary
