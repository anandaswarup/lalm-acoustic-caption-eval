"""Cross-metric and model comparison (E21). Synthetic scores and ratings only."""

import math

import numpy as np
import pytest

from src.analysis.comparison import (
    POOLED,
    concordance_rows,
    kendall_tau_b,
    metric_correlations,
    order_stability,
    ranking_rows,
    ranks,
    rating_pairs,
    report_markdown,
    score_matrix,
)
from src.evaluation.aggregate import bootstrap_indices
from src.evaluation.scoring import ScoreSet
from src.metrics.registry import Score

SYSTEMS = ["s1", "s2", "s3"]
IDS = [f"clotho_eval_{i:04d}" for i in range(60)]


def score_sets(seed: int = 0) -> list[ScoreSet]:
    """`b` is a noisy increasing function of `a`; `c` is unrelated noise."""
    rng = np.random.default_rng(seed)
    sets = []
    for offset, system in enumerate(SYSTEMS):
        a = rng.normal(offset, 1, len(IDS))
        scores = {"a": a, "b": a + rng.normal(0, 0.3, len(IDS)), "c": rng.normal(0, 1, len(IDS))}
        sets.append(ScoreSet(system, "five_reference", "all", IDS, {m: Score(0.0, v) for m, v in scores.items()}))
    return sets


def test_score_matrix_and_correlations():
    sets = score_sets()
    assert score_matrix(sets, "a").shape == (len(IDS), len(SYSTEMS))
    with pytest.raises(ValueError):
        score_matrix([sets[0], ScoreSet("x", "c", "s", IDS[:5], sets[0].scores)], "a")
    rows = metric_correlations(sets, ["a", "b", "c"], num_resamples=200, confidence=0.95, seed=1)
    assert len(rows) == 3 * (len(SYSTEMS) + 1)
    by = {(r["scope"], r["metric_a"], r["metric_b"]): r for r in rows}
    pooled = by[(POOLED, "a", "b")]
    assert pooled["num_captions"] == len(IDS) * len(SYSTEMS) and float(pooled["spearman"]) > 0.9
    assert float(pooled["ci_low"]) <= float(pooled["spearman"]) <= float(pooled["ci_high"])
    assert abs(float(by[("s1", "a", "c")]["spearman"])) < 0.4 and by[("s1", "a", "c")]["num_captions"] == len(IDS)


def test_ranks_tau_and_stability():
    assert ranks({"x": 0.1, "y": 0.3, "z": 0.3}) == {"x": 3, "y": 1, "z": 2}  # ties keep insertion order
    assert kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(2 / 3)
    assert kendall_tau_b([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(5 / math.sqrt(5 * 6))
    assert math.isnan(kendall_tau_b([1, 1, 1], [1, 2, 3]))
    clear = np.tile([1.0, 0.0], (50, 1))  # first system always higher
    assert order_stability(clear, bootstrap_indices(50, 100, 0)) == 1.0
    noisy = np.random.default_rng(0).integers(0, 2, size=(50, 2)).astype(float)
    assert 0 < order_stability(noisy, bootstrap_indices(50, 500, 0)) < 1
    rankers = {
        "m1": {"source": "E12", "values": {"s1": 0.1, "s2": 0.2, "s3": 0.3}, "stability": 0.9},
        "m2": {"source": "E19", "values": {"s1": 0.9, "s2": 0.2, "s3": 0.3}, "stability": 0.4},
    }
    rows, taus = ranking_rows(rankers, SYSTEMS)
    assert [(r["system"], r["rank"]) for r in rows if r["ranker"] == "m2"] == [("s1", 1), ("s2", 3), ("s3", 2)]
    assert taus == [{"ranker_a": "m1", "ranker_b": "m2", "kendall_tau_b": repr(-1 / 3)}]


def test_rating_pairs():
    assert rating_pairs(np.array([1, 3, 1, 2])) == [(0, 1), (0, 3), (2, 1), (3, 1), (2, 3)]
    assert rating_pairs(np.array([2, 2, 2, 2])) == []


def test_concordance():
    rng = np.random.default_rng(0)
    ratings = rng.integers(1, 4, size=(40, 4))
    scores = {
        "follows": -ratings + rng.normal(0, 0.01, ratings.shape),  # better rating, higher score
        "opposes": ratings.astype(float),
        "constant": np.zeros(ratings.shape),
    }
    rows = concordance_rows(ratings, scores, num_resamples=500, num_permutations=500, confidence=0.95, alpha=0.05,
                            seed=0)  # fmt: skip
    by = {(r["metric"], r["pairs"]): r for r in rows}
    assert float(by[("follows", "all")]["concordance"]) == 1.0 and by[("follows", "all")]["significant"]
    assert float(by[("opposes", "all")]["concordance"]) == 0.0 and by[("opposes", "all")]["significant"]
    constant = by[("constant", "all")]
    assert float(constant["concordance"]) == 0.5 and constant["ties"] == constant["num_pairs"]
    assert not constant["significant"]
    expected = sum(len(rating_pairs(r)) for r in ratings)
    assert by[("follows", "all")]["num_pairs"] == expected
    extreme = sum(sum(sorted((r[i], r[j])) == [1, 3] for i, j in rating_pairs(r)) for r in ratings)
    assert by[("follows", "1_vs_3")]["num_pairs"] == extreme
    one = by[("follows", "all")]
    assert one["concordant"] + one["ties"] + one["discordant"] == one["num_pairs"]


def test_report_markdown():
    sets = score_sets()
    rankers = {m: {"source": "E12", "values": {s.system: float(s.scores[m].items.mean()) for s in sets},
                   "stability": 1.0} for m in ("a", "b")}  # fmt: skip
    rankers["e18"] = {"source": "E18", "values": {"s1": 0.5, "s2": 0.3, "s3": 0.4}, "stability": 0.8}
    rankers["e19"] = {"source": "E19", "values": {"s1": 0.9, "s2": 0.3, "s3": 0.4}, "stability": 0.3}
    rankings, taus = ranking_rows(rankers, SYSTEMS)
    ratings = np.random.default_rng(1).integers(1, 4, size=(10, 3))
    scores = {m: score_matrix(sets, m)[:10] for m in ("a", "b")}
    report = {
        "run_info": {"git_commit": "abc", "git_dirty": False, "timestamp_utc": "now"},
        "display_names": {s: s.upper() for s in SYSTEMS},
        "metrics": ["a", "b"],
        "metric_labels": {"a": "A", "b": "B"},
        "ranker_labels": {"a": "A", "b": "B", "e18": "E18 share", "e19": "E19 share"},
        "review_rankers": ["e18", "e19"],
        "confidence": 0.95,
        "inputs": {"x": {"path": "x.csv", "sha256": "0" * 64}},
        "rankings": rankings,
        "kendall_tau": taus,
        "concordance": concordance_rows(ratings, scores, 100, 100, 0.95, 0.05, 0),
        "correlations": metric_correlations(sets, ["a", "b"], 50, 0.95, 0),
        "limitations": ["one"],
        "outputs": {"kendall_tau": "tau.csv", "correlations": "corr.csv"},
    }
    text = report_markdown(report)
    assert "| E19 share | 0.900 (1) | 0.300 (3) | 0.400 (2) | 30% |" in text
    assert "**Pairs rated 1 vs 3**" in text and "| B | 0.9" in text


def test_figure_panels(tmp_path):
    from src.analysis.figures import dot_panels, e12_panels, e14_panels
    from src.analysis.qualitative_figures import save

    summary = [{"sample": sample, "metric": "a", "system": s, "item_mean": "0.5", "ci_low": "0.4", "ci_high": "0.6"}
               for s in SYSTEMS for sample in ("all", "valid_for_all_models")]  # fmt: skip
    tests = [{"family": family, "sample": "all", "metric": "a", "system_a": s, "difference": "-0.1", "ci_low": "-0.2",
              "ci_high": "0.0", "significant": str(s != "s3")}
             for s in SYSTEMS for family in ("rq2_model_vs_human", "rq1_model_pairs")]  # fmt: skip
    [e12] = e12_panels(summary, ["a"], {"a": "A"})
    assert e12 == {"label": "A", "rows": {s: (0.5, 0.4, 0.6, True) for s in SYSTEMS}}
    [e14] = e14_panels(tests, ["a"], {"a": "A"})
    assert [e14["rows"][s][3] for s in SYSTEMS] == [True, True, False]
    names = {s: s.upper() for s in SYSTEMS}
    first = save(dot_panels([e14] * 6, SYSTEMS, names, "T", zero_line=True), tmp_path / "f", ["png"], 50)
    again = save(dot_panels([e14] * 6, SYSTEMS, names, "T", zero_line=True), tmp_path / "g", ["png"], 50)
    assert first[0].read_bytes() == again[0].read_bytes()
