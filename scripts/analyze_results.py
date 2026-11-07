"""Paired bootstrap confidence intervals and cross-metric comparisons.

E16: formal paired tests (sign-flip permutation), Holm-Bonferroni within the families of `statistics` in
configs/metrics.yaml, bootstrap CIs and paired effect sizes for the model pairs of E12, the model-human comparison of
E14 (with its sensitivity samples), the model-human length differences and the length-score correlations of E15, and
the ranking stability of E12 and E14. Reads the committed per-item tables only (no rescoring, no Java or models).
Writes outputs/scores/statistics_{tests,correlations,ranks,orders}.csv and
outputs/reports/statistical_analysis.{json,md}. E21's cross-metric comparisons are in compare_metrics.py and its
figures in generate_figures.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.analysis.lengths import association_rows, human_lengths, model_lengths  # noqa: E402
from src.analysis.statistics import (  # noqa: E402
    ASSUMPTIONS,
    CORRELATION_COLUMNS,
    MAIN_METRICS,
    ORDER_COLUMNS,
    RANK_COLUMNS,
    TEST_COLUMNS,
    Resampling,
    apply_holm,
    comparison_rows,
    correlation_rows,
    decision_changes,
    model_pairs,
    paired_test,
    rank_tables,
    report_markdown,
    sensitivity_counts,
)
from src.data.clotho import read_jsonl, sha256sum  # noqa: E402
from src.data.manifest import verify_checksums, write_json  # noqa: E402
from src.evaluation import references as conditions  # noqa: E402
from src.evaluation.inputs import load_model_captions, load_references  # noqa: E402
from src.evaluation.matched import HUMAN, load_human_baseline, matched_samples, read_overlaps  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.evaluation.scoring import read_csv, read_score_sets, write_csv  # noqa: E402
from src.evaluation.summary import ALL_RECORDINGS, POSITION_MEAN, VALID_FOR_ALL, restrict  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import MANIFEST_DIR, OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

SCORES = OUTPUT_DIR / "scores"
REPORTS = OUTPUT_DIR / "reports"
CAPTIONS = OUTPUT_DIR / "captions" / "all_model_captions.jsonl"
CAPTION_REPORT = REPORTS / "caption_validation.json"
INPUTS = {
    "five_reference_scores": SCORES / "five_reference_scores.csv",
    "five_reference_corpus": SCORES / "five_reference_corpus.csv",
    "five_reference_differences": SCORES / "five_reference_differences.csv",
    "four_reference_scores": SCORES / "four_reference_scores.csv.gz",
    "four_reference_corpus": SCORES / "four_reference_corpus.csv",
    "matched_human_model_comparison": SCORES / "matched_human_model_comparison.csv",
    "human_loo_scores": SCORES / "human_loo_scores.csv",
    "human_loo_corpus": SCORES / "human_loo_corpus.csv",
    "human_loo_overlap": SCORES / "human_loo_overlap.csv",
    "captions": CAPTIONS,
}
FAMILIES = {
    "rq1_model_pairs": "Every pair of models on every main metric, five references (E12).",
    "rq2_model_vs_human": "Every model minus the human baseline on every main metric, matched four references, "
    "five-subset mean (E14).",
    "length_vs_human": "Every model's caption length minus its recording's mean reference length (E15).",
    "length_score_correlation": "Spearman correlation of raw caption length with score, every system and "
    "association metric (E15).",
}


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--scores-dir", help="default: outputs/scores")
    parser.add_argument("--report-dir", help="default: outputs/reports")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("metrics", args.overrides)
    stats, loo = config["statistics"], config["leave_one_out"]
    association_metrics = config["lengths"]["association_metrics"]
    confidence = config["bootstrap"]["confidence"]
    resampling = Resampling(
        stats["num_resamples"], stats["num_permutations"], stats["seed"], confidence, stats["alpha"]
    )
    boot = (resampling.num_resamples, resampling.confidence, resampling.seed)

    scores_dir = resolve_path(args.scores_dir or SCORES)
    report_dir = resolve_path(args.report_dir or REPORTS)
    outputs = {
        "tests": scores_dir / "statistics_tests.csv",
        "correlations": scores_dir / "statistics_correlations.csv",
        "ranks": scores_dir / "statistics_ranks.csv",
        "orders": scores_dir / "statistics_orders.csv",
        "report_json": report_dir / "statistical_analysis.json",
        "report_md": report_dir / "statistical_analysis.md",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1
    problems = [f"manifests: {e}" for e in verify_checksums(MANIFEST_DIR / "SHA256SUMS")]
    problems += [f"captions: {e}" for e in verify_checksums(OUTPUT_DIR / "captions" / "SHA256SUMS")]
    caption_report = json.loads(CAPTION_REPORT.read_text(encoding="utf-8"))
    if caption_report["output"]["sha256"] != sha256sum(CAPTIONS):
        problems.append(f"{relative(CAPTION_REPORT)} describes a different caption file")
    for problem in problems:
        logger.error("%s", problem)
    if problems:
        return 1

    dataset, models = load_config("dataset", args.overrides), load_config("models", args.overrides)["models"]
    sample_ids = [record["sample_id"] for record in read_jsonl(resolve_path(dataset["evaluation"]["manifest"]))]
    references_path = resolve_path(dataset["evaluation"]["references"])
    references = load_references(references_path)
    valid_ids = caption_report["valid_for_all_models"]["sample_ids"]
    subsets = conditions.subsets(conditions.FOUR_REFERENCE, loo["num_positions"])

    five = {s.system: s for s in read_score_sets(INPUTS["five_reference_scores"], INPUTS["five_reference_corpus"])}
    four = read_score_sets(INPUTS["four_reference_scores"], INPUTS["four_reference_corpus"])
    if list(five) != list(models) or list(dict.fromkeys(s.system for s in four)) != list(models):
        logger.error("Score tables hold %s and %s, models.yaml %s", list(five), [s.system for s in four], list(models))
        return 1
    if any(s.sample_ids != sample_ids for s in [*five.values(), *four]):
        logger.error("Score tables do not cover the evaluation manifest in order")
        return 1
    try:
        human_sets = load_human_baseline(
            INPUTS["human_loo_scores"],
            INPUTS["human_loo_corpus"],
            REPORTS / "human_loo_evaluation.json",
            subsets,
            sample_ids,
        )
        overlaps = read_overlaps(INPUTS["human_loo_overlap"], sample_ids, loo["num_positions"])
    except ValueError as error:
        logger.error("Human baseline: %s", error)
        return 1
    samples = matched_samples(four, human_sets, valid_ids, overlaps, loo["overlap_thresholds"])
    five_sets = list(five.values())
    five_samples = [(ALL_RECORDINGS, five_sets), (VALID_FOR_ALL, restrict(five_sets, valid_ids))]
    logger.info("%d recordings, %d valid for all models; %d E14 samples", len(sample_ids), len(valid_ids), len(samples))

    tests = []
    for name, sets in five_samples:
        tests += comparison_rows("rq1_model_pairs", name, model_pairs(sets), MAIN_METRICS, resampling)
    for name, _, model_sets, human_set in samples:
        pairs = [(model_set, human_set) for model_set in model_sets]
        tests += comparison_rows("rq2_model_vs_human", name, pairs, MAIN_METRICS, resampling)
    logger.info("Score comparisons: %d tests", len(tests))

    captions = load_model_captions(CAPTIONS)
    lengths = [model_lengths(key, captions[key], references, sample_ids) for key in models]
    lengths.append(human_lengths(references, sample_ids))
    human_scores = {s.subset: s for s in human_sets}
    item_scores = {key: {m: five[key].scores[m].items[:, None] for m in association_metrics} for key in models}
    item_scores[HUMAN] = {
        m: np.stack([human_scores[s].scores[m].items for s in subsets], axis=1) for m in association_metrics
    }
    position = {sid: i for i, sid in enumerate(sample_ids)}
    associations, data = [], {}
    for name, ids in ((ALL_RECORDINGS, sample_ids), (VALID_FOR_ALL, valid_ids)):
        rows = [position[sid] for sid in ids]
        for item in (x.restrict(ids) for x in lengths):
            if item.system != HUMAN:
                test = paired_test(item.words[:, 0], item.reference_mean[:, 0], resampling)
                row = {"family": "length_vs_human", "condition": "caption_length", "subset": "", "sample": name}
                row |= {"metric": "words", "system_a": item.system, "system_b": "human_reference_mean", **test}
                tests.append({column: row[column] for column in TEST_COLUMNS})
            scores = {m: values[rows] for m, values in item_scores[item.system].items()}
            associations += association_rows(item, scores, name, *boot)
            for metric, values in scores.items():
                data[(item.system, name, "words", metric)] = (item.words, values)
                data[(item.system, name, "words_minus_reference_mean", metric)] = (
                    item.words - item.reference_mean,
                    values,
                )
        logger.info("Lengths and correlations done on %s", name)
    correlations = correlation_rows("length_score_correlation", associations, data, resampling)

    apply_holm(tests, lambda row: row["sample"] == ALL_RECORDINGS, resampling.alpha)
    apply_holm(
        correlations,
        lambda row: row["sample"] == ALL_RECORDINGS and row["predictor"] == "words",
        resampling.alpha,
    )

    ranks, orders = [], []
    for name, sets in five_samples:
        r, o = rank_tables(sets, name, MAIN_METRICS, resampling)
        ranks, orders = ranks + r, orders + o
    for name, _, model_sets, human_set in samples:
        r, o = rank_tables([*model_sets, human_set], name, MAIN_METRICS, resampling)
        ranks, orders = ranks + r, orders + o

    earlier = {
        "rq1_model_pairs": [r for r in read_csv(INPUTS["five_reference_differences"]) if r["sample"] == ALL_RECORDINGS],
        "rq2_model_vs_human": [
            r for r in read_csv(INPUTS["matched_human_model_comparison"]) if r["sample"] == ALL_RECORDINGS
        ],
    }
    words_only = [r for r in correlations if r["predictor"] == "words"]
    sensitivity = sensitivity_counts(tests, ALL_RECORDINGS, lambda r: (r["metric"], r["system_a"], r["system_b"]))
    sensitivity += sensitivity_counts(words_only, ALL_RECORDINGS, lambda r: (r["system"], r["metric"]))

    write_csv(tests, outputs["tests"], TEST_COLUMNS)
    write_csv(correlations, outputs["correlations"], CORRELATION_COLUMNS)
    write_csv(ranks, outputs["ranks"], RANK_COLUMNS)
    write_csv(orders, outputs["orders"], ORDER_COLUMNS)
    primary = [*[r for r in tests if r["p_holm"] != ""], *[r for r in correlations if r["p_holm"] != ""]]
    report = {
        "run_info": run_info,
        "task": "E16",
        "display_names": {key: model["display_name"] for key, model in models.items()},
        "settings": {**stats, "confidence": resampling.confidence},
        "families": FAMILIES,
        "counts": {family: sum(r["family"] == family for r in primary) for family in FAMILIES},
        "primary_sample": ALL_RECORDINGS,
        "metrics": MAIN_METRICS,
        "association_metrics": association_metrics,
        "inputs": {
            **{key: {"path": relative(path), "sha256": sha256sum(path)} for key, path in INPUTS.items()},
            "references": {"path": relative(references_path), "sha256": sha256sum(references_path)},
        },
        "assumptions": ASSUMPTIONS,
        "tests": tests,
        "correlations": correlations,
        "ranks": ranks,
        "orders": orders,
        "rank_titles": [
            [conditions.FIVE_REFERENCE, "all", "Models, five references (E12)"],
            [conditions.FOUR_REFERENCE, POSITION_MEAN, "Models and human, matched four references (E14)"],
        ],
        "earlier": {
            "rq1_model_pairs": f"`{relative(INPUTS['five_reference_differences'])}` (E12)",
            "rq2_model_vs_human": f"`{relative(INPUTS['matched_human_model_comparison'])}` (E14)",
        },
        "decision_changes": {
            family: decision_changes(tests, rows, family, ALL_RECORDINGS) for family, rows in earlier.items()
        },
        "sensitivity": sensitivity,
        "outputs": {name: relative(path) for name, path in outputs.items()},
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    for family, count in report["counts"].items():
        significant = sum(bool(r["significant"]) for r in primary if r["family"] == family)
        logger.info("%s: %d of %d significant after Holm", family, significant, count)
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
