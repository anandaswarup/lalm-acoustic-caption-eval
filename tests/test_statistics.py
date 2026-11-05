"""Permutation tests, Holm correction, effect sizes and ranking stability (E16). No Java or models."""

import math

import numpy as np
import pytest

from src.analysis.statistics import (
    Resampling,
    apply_holm,
    comparison_rows,
    effect_size,
    holm,
    model_pairs,
    permutation_spearman_p,
    rank_tables,
    sensitivity_counts,
    sign_flip_p,
)
from src.evaluation.scoring import ScoreSet
from src.metrics.registry import Score

IDS = [f"s{i}" for i in range(60)]


def resampling(n: int = 2000) -> Resampling:
    return Resampling(num_resamples=n, num_permutations=n, seed=42, confidence=0.95, alpha=0.05)


def make_sets(gaps: dict[str, float], seed: int = 0) -> list[ScoreSet]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.5, 0.1, len(IDS))
    sets = []
    for system, gap in gaps.items():
        items = noise + gap + rng.normal(0, 0.02, len(IDS))
        sets.append(ScoreSet(system, "five_reference", "all", list(IDS), {"cider_d": Score(0.0, items)}))
    return sets


def test_holm_matches_hand_computation():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]).max() <= 1.0


def test_sign_flip_p_bounds():
    signs = resampling().signs(len(IDS))
    assert sign_flip_p(np.zeros(len(IDS)), signs) == 1.0
    shifted = np.random.default_rng(0).normal(1.0, 0.1, len(IDS))
    assert sign_flip_p(shifted, signs) == pytest.approx(1 / (len(signs) + 1))
    noise = np.random.default_rng(1).normal(0.0, 1.0, len(IDS))
    assert sign_flip_p(noise, signs) > 0.05


def test_effect_size():
    assert effect_size(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    assert math.isnan(effect_size(np.ones(5)))


def test_comparisons_and_holm_only_on_the_primary_sample():
    res = resampling()
    sets = make_sets({"a": 0.0, "b": 0.0, "c": 0.2})
    rows = comparison_rows("fam", "all", model_pairs(sets), ["cider_d"], res)
    rows += comparison_rows("fam", "other", model_pairs(sets), ["cider_d"], res)
    apply_holm(rows, lambda row: row["sample"] == "all", res.alpha)
    primary = {(r["system_a"], r["system_b"]): r for r in rows if r["sample"] == "all"}
    assert primary[("a", "c")]["significant"] is True and float(primary[("a", "c")]["difference"]) < 0
    assert primary[("a", "b")]["significant"] is False
    assert float(primary[("a", "b")]["p_holm"]) >= float(primary[("a", "b")]["p_value"])
    assert all(r["p_holm"] == "" for r in rows if r["sample"] == "other")
    counts = sensitivity_counts(rows, "all", lambda r: (r["system_a"], r["system_b"]))
    assert counts == [
        {"family": "fam", "sample": "other", "num_tests": 3, "significant_uncorrected": 2, "differs_from_primary": []}
    ]
    with pytest.raises(ValueError):
        comparison_rows("fam", "all", [(sets[0], ScoreSet("x", "c", "s", IDS[:5], sets[0].scores))], ["cider_d"], res)


def test_permutation_spearman_keeps_recordings_together():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 5))
    perms = resampling().permutations(60)
    assert permutation_spearman_p(x, x + rng.normal(0, 0.3, x.shape), perms) == pytest.approx(1 / (len(perms) + 1))
    assert permutation_spearman_p(x, rng.normal(size=x.shape), perms) > 0.01
    assert sorted(perms[0].tolist()) == list(range(60))


def test_rank_tables_shares():
    sets = make_sets({"a": 0.0, "b": 0.3, "c": 0.001})
    ranks, orders = rank_tables(sets, "all", ["cider_d"], resampling())
    assert orders[0]["observed_order"].split(";")[0] == "b"
    for system in "abc":
        shares = [float(r["share"]) for r in ranks if r["system"] == system]
        assert sum(shares) == pytest.approx(1.0)
    first = {r["system"]: float(r["share"]) for r in ranks if r["rank"] == 1}
    assert first["b"] == 1.0
    assert 0 < float(orders[0]["share_observed_order"]) < 1  # a and c are too close to order reliably
    with pytest.raises(ValueError):
        rank_tables(sets, "all", ["fluency_error"], resampling())
