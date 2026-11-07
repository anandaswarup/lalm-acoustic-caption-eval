"""Check, unblind and summarize the audio-grounded review.

E18 Part 3: run once both templates (audio_annotations_reviewer_1.csv, disputed_verification_reviewer_1.csv) are final.
Checks that every caption has a rating and every disputed phrase a verdict and that no fixed column was changed,
verifies the blinding key against sample_manifest.json and the review inputs against audio_review_manifest.json, then
writes the unblinded captions and phrases, the rating and verdict shares per model (Wilson intervals), the E17 category
x audio rating cross-tab, the category-6 split and the mean five-reference scores per rating to outputs/qualitative/,
and outputs/reports/audio_review.{json,md}. Labels are not changed after unblinding.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.annotations import (  # noqa: E402
    check_annotations,
    parse_category,
    read_table,
    scores_by_category,
    unblind,
)
from src.analysis.audio_review import (  # noqa: E402
    AUDIO_LIMITATIONS,
    CAPTION_COLUMNS,
    CROSSTAB_COLUMNS,
    ITEM_COLUMNS,
    MISMATCH_COLUMNS,
    PHRASE_REVIEW_COLUMNS,
    PHRASE_ROW_COLUMNS,
    RATING_SUMMARY_COLUMNS,
    SCORE_COLUMNS,
    UNSUPPORTED_COLUMNS,
    VERDICT_SUMMARY_COLUMNS,
    audio_report_markdown,
    audio_template,
    check_labels,
    crosstab,
    mismatch_split,
    parse_verdict,
    phrase_template,
    rating_conflicts,
    share_summary,
    unblind_audio,
    unsupported_summary,
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
    parser.add_argument("--report-dir", help="default: outputs/reports")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("qualitative", args.overrides)
    if config["reviewers"] != 1:
        logger.error("Only one reviewer is supported (configs/qualitative.yaml has %s)", config["reviewers"])
        return 1
    settings = config["audio_review"]
    categories = [category["code"] for category in config["categories"]]
    ratings = [rating["code"] for rating in settings["ratings"]]
    verdicts = [verdict["code"] for verdict in settings["verdicts"]]

    out_dir = resolve_path(config["output_dir"])
    report_dir = resolve_path(args.report_dir or OUTPUT_DIR / "reports")
    paths = {
        "audio_annotations": out_dir / "audio_annotations_reviewer_1.csv",
        "phrase_verdicts": out_dir / "disputed_verification_reviewer_1.csv",
        "review": out_dir / "review_items.csv",
        "annotations": out_dir / "annotations_reviewer_1.csv",
        "sample_manifest": out_dir / "sample_manifest.json",
        "audio_review_manifest": out_dir / "audio_review_manifest.json",
        "guidelines": out_dir / "audio_review_guidelines.md",
    }
    key_path = out_dir / "blinding_key.csv"
    outputs = {
        "captions": out_dir / "audio_review_annotations.csv",
        "phrases": out_dir / "disputed_verification.csv",
        "rating_summary": out_dir / "audio_rating_summary.csv",
        "verdict_summary": out_dir / "verdict_summary.csv",
        "unsupported_summary": out_dir / "unsupported_summary.csv",
        "crosstab": out_dir / "category_by_audio_rating.csv",
        "mismatch_split": out_dir / "category_6_split.csv",
        "scores_by_audio_rating": out_dir / "scores_by_audio_rating.csv",
        "report_json": report_dir / "audio_review.json",
        "report_md": report_dir / "audio_review.md",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1

    sample_manifest = json.loads(paths["sample_manifest"].read_text(encoding="utf-8"))
    if not key_path.is_file() or sha256sum(key_path) != sample_manifest["key"]["sha256"]:
        logger.error("%s is missing or does not match the SHA-256 in sample_manifest.json", relative(key_path))
        return 1
    audio_manifest = json.loads(paths["audio_review_manifest"].read_text(encoding="utf-8"))
    changed = [
        item["path"]
        for item in audio_manifest["inputs"].values()
        if sha256sum(resolve_path(item["path"])) != item["sha256"]
    ]
    if changed:
        logger.error("Changed since the review material was prepared (audio_review_manifest.json): %s", changed)
        return 1

    review, annotations = read_table(paths["review"]), read_table(paths["annotations"])
    if errors := check_annotations(annotations, review, set(categories)):
        logger.error("The E17 annotations do not check (%d problems, first: %s)", len(errors), errors[0])
        return 1
    audio, phrases = read_table(paths["audio_annotations"]), read_table(paths["phrase_verdicts"])
    errors = check_labels(
        audio, audio_template(review), ITEM_COLUMNS, "audio_rating", lambda v: parse_category(v, set(ratings))
    )
    errors += check_labels(
        phrases,
        phrase_template(annotations),
        PHRASE_REVIEW_COLUMNS,
        "verdict",
        lambda v: parse_verdict(v, set(verdicts)),
    )
    if errors:
        for error in errors[:50]:
            logger.error("%s", error)
        logger.error("%d problems in the E18 templates; nothing unblinded", len(errors))
        return 1
    captions, phrase_rows = unblind_audio(unblind(annotations, read_table(key_path)), audio, phrases)
    logger.info("%d captions and %d phrases checked and unblinded", len(captions), len(phrase_rows))

    models = load_config("models", args.overrides)["models"]
    if sorted({row["model_key"] for row in captions}) != sorted(models):
        logger.error("Reviewed models %s differ from models.yaml", sorted({row["model_key"] for row in captions}))
        return 1
    metrics, confidence, keys = settings["score_metrics"], settings["confidence"], list(models)
    scores = {}
    for score_set in read_score_sets(*SCORES):
        for index, sample_id in enumerate(score_set.sample_ids):
            scores[(score_set.system, sample_id)] = {m: float(score_set.scores[m].items[index]) for m in metrics}

    results = {
        "rating_summary": share_summary(captions, keys, "audio_rating", ratings, confidence),
        "verdict_summary": share_summary(phrase_rows, keys, "verdict", verdicts, confidence),
        "unsupported_summary": unsupported_summary(captions, keys, confidence),
        "crosstab": crosstab(captions, keys, categories, ratings),
        "mismatch_split": mismatch_split(captions, keys),
        "scores_by_audio_rating": scores_by_category(captions, scores, metrics, ratings, field="audio_rating"),
    }
    columns = {
        "rating_summary": RATING_SUMMARY_COLUMNS,
        "verdict_summary": VERDICT_SUMMARY_COLUMNS,
        "unsupported_summary": UNSUPPORTED_COLUMNS,
        "crosstab": CROSSTAB_COLUMNS,
        "mismatch_split": MISMATCH_COLUMNS,
        "scores_by_audio_rating": SCORE_COLUMNS,
    }
    write_csv(captions, outputs["captions"], CAPTION_COLUMNS)
    write_csv(phrase_rows, outputs["phrases"], PHRASE_ROW_COLUMNS)
    for name, rows in results.items():
        write_csv(rows, outputs[name], columns[name])
    conflicts = rating_conflicts(captions)
    if conflicts:
        logger.warning("%d captions rated 1 have an unsupported phrase (reported, not changed)", len(conflicts))
    report = {
        "run_info": run_info,
        "task": "E18",
        "display_names": {key: model["display_name"] for key, model in models.items()},
        "categories": config["categories"],
        "ratings": settings["ratings"],
        "verdicts": settings["verdicts"],
        "confidence": confidence,
        "metrics": metrics,
        "metric_labels": {m: BY_NAME[m].label for m in metrics},
        "num_items": len(captions),
        "num_recordings": len({row["sample_id"] for row in captions}),
        "num_phrases": len(phrase_rows),
        "items_per_model": len(captions) // len(models),
        "inputs_verified": True,
        "inputs": {
            **{name: {"path": relative(path), "sha256": sha256sum(path)} for name, path in paths.items()},
            "key": {"path": relative(key_path), "sha256": sha256sum(key_path)},
        },
        **results,
        "rating_conflicts": conflicts,
        "limitations": AUDIO_LIMITATIONS,
        "outputs": {name: relative(path) for name, path in outputs.items()},
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(audio_report_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
