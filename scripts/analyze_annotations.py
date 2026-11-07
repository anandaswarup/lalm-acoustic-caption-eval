"""Check, unblind and summarize the qualitative annotations.

E17: run once the reviewer's labels in outputs/qualitative/annotations_reviewer_1.csv are final. Checks that every item
has a valid category and that no review column was changed, verifies the blinding key against the SHA-256 in
sample_manifest.json, then writes the unblinded annotations, the category shares per model (Wilson intervals), the
disputed-detail counts and the mean five-reference scores per category to outputs/qualitative/, and
outputs/reports/qualitative_annotation.{json,md}. Labels are not changed after unblinding.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.annotations import (  # noqa: E402
    ANNOTATED_COLUMNS,
    DISPUTED_COLUMNS,
    LIMITATIONS,
    SCORE_COLUMNS,
    SUMMARY_COLUMNS,
    category_summary,
    check_annotations,
    disputed_summary,
    read_table,
    report_markdown,
    scores_by_category,
    unblind,
)
from src.data.clotho import sha256sum  # noqa: E402
from src.data.manifest import write_json  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.evaluation.scoring import read_score_sets, write_csv  # noqa: E402
from src.metrics.registry import BY_NAME  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

SCORES = (OUTPUT_DIR / "scores" / "five_reference_scores.csv", OUTPUT_DIR / "scores" / "five_reference_corpus.csv")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--annotations", help="completed template (default: annotations_reviewer_1.csv)")
    parser.add_argument("--report-dir", help="default: outputs/reports")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("qualitative", args.overrides)
    if config["reviewers"] != 1:
        logger.error("Only one reviewer is supported (configs/qualitative.yaml has %s)", config["reviewers"])
        return 1
    settings = config["annotation"]
    codes = [category["code"] for category in config["categories"]]

    out_dir = resolve_path(config["output_dir"])
    report_dir = resolve_path(args.report_dir or OUTPUT_DIR / "reports")
    paths = {
        "annotations": resolve_path(args.annotations or out_dir / "annotations_reviewer_1.csv"),
        "review": out_dir / "review_items.csv",
        "manifest": out_dir / "sample_manifest.json",
        "guidelines": out_dir / "annotation_guidelines.md",
    }
    key_path = out_dir / "blinding_key.csv"
    outputs = {
        "annotated": out_dir / "qualitative_annotations.csv",
        "summary": out_dir / "category_summary.csv",
        "disputed": out_dir / "disputed_summary.csv",
        "scores_by_category": out_dir / "scores_by_category.csv",
        "report_json": report_dir / "qualitative_annotation.json",
        "report_md": report_dir / "qualitative_annotation.md",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1

    annotations, review = read_table(paths["annotations"]), read_table(paths["review"])
    if errors := check_annotations(annotations, review, set(codes)):
        for error in errors[:50]:
            logger.error("%s", error)
        logger.error("%d problems in %s; nothing unblinded", len(errors), relative(paths["annotations"]))
        return 1
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if not key_path.is_file() or sha256sum(key_path) != manifest["key"]["sha256"]:
        manifest_path = relative(paths["manifest"])
        logger.error("%s is missing or does not match the SHA-256 in %s", relative(key_path), manifest_path)
        return 1
    rows = unblind(annotations, read_table(key_path))
    logger.info("%d items checked and unblinded", len(rows))

    models = load_config("models", args.overrides)["models"]
    if sorted({row["model_key"] for row in rows}) != sorted(models):
        logger.error("Annotated models %s differ from models.yaml", sorted({row["model_key"] for row in rows}))
        return 1
    metrics = settings["score_metrics"]
    scores = {}
    for score_set in read_score_sets(*SCORES):
        for index, sample_id in enumerate(score_set.sample_ids):
            scores[(score_set.system, sample_id)] = {m: float(score_set.scores[m].items[index]) for m in metrics}

    summary = category_summary(rows, list(models), codes, settings["confidence"])
    disputed = disputed_summary(rows, list(models))
    by_category = scores_by_category(rows, scores, metrics, codes)
    write_csv(rows, outputs["annotated"], ANNOTATED_COLUMNS)
    write_csv(summary, outputs["summary"], SUMMARY_COLUMNS)
    write_csv(disputed, outputs["disputed"], DISPUTED_COLUMNS)
    write_csv(by_category, outputs["scores_by_category"], SCORE_COLUMNS)
    report = {
        "run_info": run_info,
        "task": "E17",
        "display_names": {key: model["display_name"] for key, model in models.items()},
        "categories": config["categories"],
        "confidence": settings["confidence"],
        "metrics": metrics,
        "metric_labels": {m: BY_NAME[m].label for m in metrics},
        "num_items": len(rows),
        "num_recordings": len({row["sample_id"] for row in rows}),
        "items_per_model": len(rows) // len(models),
        "key_verified": True,
        "inputs": {
            **{name: {"path": relative(path), "sha256": sha256sum(path)} for name, path in paths.items()},
            "key": {"path": relative(key_path), "sha256": sha256sum(key_path)},
        },
        "summary": summary,
        "disputed": disputed,
        "scores_by_category": by_category,
        "limitations": LIMITATIONS,
        "outputs": {name: relative(path) for name, path in outputs.items()},
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
