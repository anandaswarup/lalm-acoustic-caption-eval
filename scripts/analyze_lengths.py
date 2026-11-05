"""Caption-length statistics for model and human captions.

E15: word counts (whitespace tokens) of the 4,180 scored model captions and the 5,225 human references; statistics per
system; model-to-human mean length ratios and per-recording differences from the reference mean, with bootstrap CIs;
and Spearman correlations between length and score, from the per-item five-reference model scores (E12) and human
leave-one-out scores (E13), for `lengths.association_metrics`. Everything is also computed on the 1,024 recordings
where every model caption is valid. Reads frozen files only (no rescoring, no Java or models). Writes
outputs/scores/caption_lengths_{per_caption,summary,associations}.csv and
outputs/reports/caption_length_analysis.{json,md}. Figures are drawn in E21 from the per-caption table.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.analysis.lengths import (  # noqa: E402
    ASSOCIATION_COLUMNS,
    HUMAN,
    LIMITATIONS,
    PER_CAPTION_COLUMNS,
    SUMMARY_COLUMNS,
    association_rows,
    human_lengths,
    model_lengths,
    per_caption_rows,
    report_markdown,
    summary_rows,
)
from src.data.clotho import read_jsonl, sha256sum  # noqa: E402
from src.data.manifest import verify_checksums, write_json  # noqa: E402
from src.evaluation.inputs import load_model_captions, load_references  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.evaluation.scoring import read_score_sets, write_csv  # noqa: E402
from src.evaluation.summary import ALL_RECORDINGS, VALID_FOR_ALL  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import MANIFEST_DIR, OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

CAPTIONS = OUTPUT_DIR / "captions" / "all_model_captions.jsonl"
CAPTION_REPORT = OUTPUT_DIR / "reports" / "caption_validation.json"
SCORES = OUTPUT_DIR / "scores"
MODEL_SCORES = (SCORES / "five_reference_scores.csv", SCORES / "five_reference_corpus.csv")
HUMAN_SCORES = (SCORES / "human_loo_scores.csv", SCORES / "human_loo_corpus.csv")
WORD_DEFINITION = "whitespace tokens of the caption (`caption.split()`), as in the dataset and completion reports."


def file_info(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256sum(path)}


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--scores-dir", help="default: outputs/scores")
    parser.add_argument("--report-dir", help="default: outputs/reports")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("metrics", args.overrides)
    bootstrap, metrics = config["bootstrap"], config["lengths"]["association_metrics"]
    boot = (bootstrap["num_resamples"], bootstrap["confidence"], bootstrap["seed"])

    scores_dir = resolve_path(args.scores_dir or SCORES)
    report_dir = resolve_path(args.report_dir or OUTPUT_DIR / "reports")
    outputs = {
        "per_caption": scores_dir / "caption_lengths_per_caption.csv",
        "summary": scores_dir / "caption_lengths_summary.csv",
        "associations": scores_dir / "caption_lengths_associations.csv",
        "report_json": report_dir / "caption_length_analysis.json",
        "report_md": report_dir / "caption_length_analysis.md",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1
    problems = [f"manifests: {e}" for e in verify_checksums(MANIFEST_DIR / "SHA256SUMS")]
    problems += [f"captions: {e}" for e in verify_checksums(OUTPUT_DIR / "captions" / "SHA256SUMS")]
    for problem in problems:
        logger.error("%s", problem)
    if problems:
        return 1

    dataset, models = load_config("dataset", args.overrides), load_config("models", args.overrides)["models"]
    captions = load_model_captions(CAPTIONS)
    if list(captions) != list(models):
        logger.error("Caption models %s do not match models.yaml %s", list(captions), list(models))
        return 1
    caption_report = json.loads(CAPTION_REPORT.read_text(encoding="utf-8"))
    if caption_report["output"]["sha256"] != sha256sum(CAPTIONS):
        logger.error("%s describes a different caption file", relative(CAPTION_REPORT))
        return 1
    references_path = resolve_path(dataset["evaluation"]["references"])
    references = load_references(references_path)
    sample_ids = [record["sample_id"] for record in read_jsonl(resolve_path(dataset["evaluation"]["manifest"]))]
    valid_ids = caption_report["valid_for_all_models"]["sample_ids"]
    positions = {len(references[sid]) for sid in sample_ids}
    if len(positions) != 1:
        logger.error("Recordings have different numbers of references: %s", sorted(positions))
        return 1
    logger.info("%d recordings x %d models; %d with every caption valid", len(sample_ids), len(models), len(valid_ids))

    lengths = [model_lengths(key, captions[key], references, sample_ids) for key in models]
    lengths.append(human_lengths(references, sample_ids))

    # Per-caption scores shaped like the lengths: one column for a model, one per held-out position for humans.
    model_sets = {s.system: s for s in read_score_sets(*MODEL_SCORES) if s.condition == "five_reference"}
    human_sets = {s.subset: s for s in read_score_sets(*HUMAN_SCORES) if s.system == HUMAN}
    subsets = [f"drop_{k}" for k in range(1, lengths[-1].words.shape[1] + 1)]
    if set(model_sets) != set(models) or set(human_sets) != set(subsets):
        logger.error("Score tables hold %s and %s", sorted(model_sets), sorted(human_sets))
        return 1
    for score_set in [*model_sets.values(), *human_sets.values()]:
        if score_set.sample_ids != sample_ids or not set(metrics) <= set(score_set.scores):
            logger.error("%s %s: other recordings or missing metrics", score_set.system, score_set.subset)
            return 1
    scores = {key: {m: model_sets[key].scores[m].items[:, None] for m in metrics} for key in models}
    scores[HUMAN] = {m: np.stack([human_sets[s].scores[m].items for s in subsets], axis=1) for m in metrics}

    position = {sid: i for i, sid in enumerate(sample_ids)}
    summary, associations = [], []
    for sample, ids in ((ALL_RECORDINGS, sample_ids), (VALID_FOR_ALL, valid_ids)):
        subset = [item.restrict(ids) for item in lengths]
        rows = [position[sid] for sid in ids]
        summary += summary_rows(subset, sample, *boot)
        for item in subset:
            item_scores = {m: values[rows] for m, values in scores[item.system].items()}
            associations += association_rows(item, item_scores, sample, *boot)
    for row in summary:
        if row["sample"] == ALL_RECORDINGS:
            ratio = row["ratio_to_human"] or "n/a"
            logger.info("%s: mean %.2f words, ratio to human %s", row["system"], float(row["mean"]), ratio)

    write_csv(per_caption_rows(lengths), outputs["per_caption"], PER_CAPTION_COLUMNS)
    write_csv(summary, outputs["summary"], SUMMARY_COLUMNS)
    write_csv(associations, outputs["associations"], ASSOCIATION_COLUMNS)
    report = {
        "run_info": run_info,
        "task": "E15",
        "display_names": {key: model["display_name"] for key, model in models.items()},
        "word_definition": WORD_DEFINITION,
        "samples": {"all": ALL_RECORDINGS, "valid": VALID_FOR_ALL},
        "association_metrics": metrics,
        "inputs": {
            "recordings": len(sample_ids),
            "valid_for_all_models": len(valid_ids),
            "references_per_recording": positions.pop(),
            "captions": file_info(CAPTIONS),
            "references": file_info(references_path),
            "model_scores": file_info(MODEL_SCORES[0]),
            "human_scores": file_info(HUMAN_SCORES[0]),
        },
        "bootstrap": bootstrap,
        "summary": summary,
        "associations": associations,
        "outputs": {name: relative(path) for name, path in outputs.items()},
        "limitations": LIMITATIONS,
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
