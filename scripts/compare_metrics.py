"""Cross-metric and model comparison.

E21: from the committed per-item scores (E12), the E16 ranking stability and the unblinded E18/E19 labels, writes the
metric x metric correlations, the model rankings by every metric and by the two reviews (with Kendall's tau-b between
rankings), and the within-recording concordance of each metric with the E19 audio ratings to outputs/scores/, and
outputs/reports/cross_metric_comparison.{json,md}. Nothing is rescored. Settings: `comparison` in configs/metrics.yaml.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.annotations import read_table  # noqa: E402
from src.analysis.comparison import (  # noqa: E402
    CONCORDANCE_COLUMNS,
    CORRELATION_COLUMNS,
    LIMITATIONS,
    RANKING_COLUMNS,
    TAU_COLUMNS,
    concordance_rows,
    metric_correlations,
    order_stability,
    ranking_rows,
    ranks,
    report_markdown,
    score_matrix,
)
from src.analysis.statistics import MAIN_METRICS  # noqa: E402
from src.data.clotho import sha256sum  # noqa: E402
from src.data.manifest import write_json  # noqa: E402
from src.evaluation.aggregate import bootstrap_indices  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.evaluation.scoring import read_score_sets, write_csv  # noqa: E402
from src.metrics.registry import BY_NAME  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

SCORES = OUTPUT_DIR / "scores"
QUALITATIVE = OUTPUT_DIR / "qualitative"
INPUTS = {
    "five_reference_scores": SCORES / "five_reference_scores.csv",
    "five_reference_corpus": SCORES / "five_reference_corpus.csv",
    "statistics_orders": SCORES / "statistics_orders.csv",
    "audio_review": QUALITATIVE / "audio_review_annotations.csv",
}
E18, E19 = "e18_categories_1_3", "e19_rated_accurate"
AGREEING_CATEGORIES = {1, 2, 3}  # agreement with the references' main content (E18)


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    settings = load_config("metrics", args.overrides)["comparison"]
    outputs = {
        "correlations": SCORES / "cross_metric_correlations.csv",
        "rankings": SCORES / "model_rankings.csv",
        "kendall_tau": SCORES / "ranking_kendall_tau.csv",
        "concordance": SCORES / "audio_rating_concordance.csv",
        "report_json": OUTPUT_DIR / "reports" / "cross_metric_comparison.json",
        "report_md": OUTPUT_DIR / "reports" / "cross_metric_comparison.md",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1
    if missing := [relative(path) for path in INPUTS.values() if not path.is_file()]:
        logger.error("Missing inputs: %s", missing)
        return 1

    models = load_config("models", args.overrides)["models"]
    systems = list(models)
    by_system = {s.system: s for s in read_score_sets(INPUTS["five_reference_scores"], INPUTS["five_reference_corpus"])}
    if sorted(by_system) != sorted(systems):
        logger.error("Scored systems %s differ from models.yaml", sorted(by_system))
        return 1
    score_sets = [by_system[system] for system in systems]
    sample_ids = score_sets[0].sample_ids
    seed, confidence = settings["seed"], settings["confidence"]
    metrics = settings["metrics"]

    logger.info("Metric correlations: %d metrics, %d resamples", len(metrics), settings["correlation_resamples"])
    correlations = metric_correlations(score_sets, metrics, settings["correlation_resamples"], confidence, seed)

    orders = {row["metric"]: row for row in read_table(INPUTS["statistics_orders"])
              if (row["condition"], row["subset"], row["sample"]) == ("five_reference", "all", "all")}  # fmt: skip
    rankers = {}
    for metric in MAIN_METRICS:
        values = {system: float(by_system[system].scores[metric].items.mean()) for system in systems}
        observed = ";".join(sorted(values, key=lambda s: ranks(values)[s]))
        if orders[metric]["observed_order"] != observed:
            logger.error("%s: order %s differs from statistics_orders.csv (%s)", metric, observed,
                         orders[metric]["observed_order"])  # fmt: skip
            return 1
        stability = float(orders[metric]["share_observed_order"])
        rankers[metric] = {"source": "E12", "values": values, "stability": stability}

    review = read_table(INPUTS["audio_review"])
    codes = list(dict.fromkeys(row["recording_code"] for row in review))
    cell = {(row["recording_code"], row["model_key"]): row for row in review}
    if len(cell) != len(codes) * len(systems):
        logger.error("The audio-review table must hold one caption per model for every recording")
        return 1
    category = np.array([[int(cell[(c, s)]["category"]) for s in systems] for c in codes])
    rating = np.array([[int(cell[(c, s)]["audio_rating"]) for s in systems] for c in codes])
    indices = bootstrap_indices(len(codes), settings["num_resamples"], seed)
    for name, indicator in ((E18, np.isin(category, list(AGREEING_CATEGORIES))), (E19, rating == 1)):
        per_recording = indicator.astype(np.float64)
        values = dict(zip(systems, per_recording.mean(axis=0).tolist(), strict=True))
        rankers[name] = {"source": "E18" if name == E18 else "E19", "values": values,
                         "stability": order_stability(per_recording, indices)}  # fmt: skip
    rankings, taus = ranking_rows(rankers, systems)

    position = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    sampled = [position[cell[(c, systems[0])]["sample_id"]] for c in codes]
    if any(cell[(c, s)]["sample_id"] != cell[(c, systems[0])]["sample_id"] for c in codes for s in systems):
        logger.error("Captions of one recording code have different sample IDs")
        return 1
    scores = {metric: score_matrix(score_sets, metric)[sampled] for metric in metrics}
    logger.info("Concordance: %d recordings, %d permutations", len(codes), settings["num_permutations"])
    concordance = concordance_rows(rating, scores, settings["num_resamples"], settings["num_permutations"], confidence,
                                   settings["alpha"], seed)  # fmt: skip

    write_csv(correlations, outputs["correlations"], CORRELATION_COLUMNS)
    write_csv(rankings, outputs["rankings"], RANKING_COLUMNS)
    write_csv(taus, outputs["kendall_tau"], TAU_COLUMNS)
    write_csv(concordance, outputs["concordance"], CONCORDANCE_COLUMNS)
    labels = {metric: BY_NAME[metric].label for metric in MAIN_METRICS}
    report = {
        "run_info": run_info,
        "task": "E21",
        "settings": settings,
        "confidence": confidence,
        "display_names": {key: model["display_name"] for key, model in models.items()},
        "metrics": metrics,
        "metric_labels": {metric: BY_NAME[metric].label for metric in MAIN_METRICS},
        "ranker_labels": {**labels, E18: "E18: categories 1-3 (sample)", E19: "E19: rated accurate (sample)"},
        "review_rankers": [E18, E19],
        "num_recordings": len(sample_ids),
        "num_sampled_recordings": len(codes),
        "inputs": {name: {"path": relative(path), "sha256": sha256sum(path)} for name, path in INPUTS.items()},
        "rankings": rankings,
        "kendall_tau": taus,
        "concordance": concordance,
        "correlations": correlations,
        "limitations": LIMITATIONS,
        "outputs": {name: relative(path) for name, path in outputs.items()},
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
