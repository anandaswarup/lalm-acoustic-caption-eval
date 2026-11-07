"""Blinded qualitative review sample and annotation templates.

E17: draws `sample.num_recordings` evaluation recordings at random (configs/qualitative.yaml, independent of model
scores), then writes to `output_dir` the blinded review sheet (one row per model caption: recording code, label A-D,
the five references and the caption as scored) and a Markdown reading view of it, one blank annotation template per
reviewer, the category list, the
sample manifest and the blinding key (git-ignored until the E18 labels are final; its SHA-256 is in the manifest).
Also writes outputs/reports/qualitative_sample.{json,md} with a descriptive comparison of the sample with all
recordings.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.analysis.annotations import read_table  # noqa: E402
from src.analysis.lengths import word_count  # noqa: E402
from src.analysis.qualitative import (  # noqa: E402
    ANNOTATION_COLUMNS,
    CATEGORY_COLUMNS,
    KEY_COLUMNS,
    LIMITATIONS,
    REVIEW_COLUMNS,
    annotation_template,
    blind,
    blinding_problems,
    has_annotations,
    report_markdown,
    review_rows,
    review_sheet_markdown,
    sample_comparison,
    select_sample,
)
from src.data.clotho import read_jsonl, sha256sum  # noqa: E402
from src.data.manifest import verify_checksums, write_json  # noqa: E402
from src.evaluation.inputs import load_references  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.evaluation.scoring import read_score_sets, write_csv  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import MANIFEST_DIR, OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

CAPTIONS = OUTPUT_DIR / "captions" / "all_model_captions.jsonl"
SCORES = (OUTPUT_DIR / "scores" / "five_reference_scores.csv", OUTPUT_DIR / "scores" / "five_reference_corpus.csv")
SCORE_METRICS = ("cider_d", "spice", "fense", "bertscore_f1_rescaled", "clap_similarity", "mace")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--output-dir", help="default: output_dir in configs/qualitative.yaml")
    parser.add_argument("--report-dir", help="default: outputs/reports")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument(
        "--keep-labelled",
        action="store_true",
        help="leave templates that hold labels untouched (after checking their review columns) and write the rest",
    )
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("qualitative", args.overrides)
    num_recordings, seed = config["sample"]["num_recordings"], config["sample"]["seed"]

    out_dir = resolve_path(args.output_dir or config["output_dir"])
    report_dir = resolve_path(args.report_dir or OUTPUT_DIR / "reports")
    templates = {n: out_dir / f"annotations_reviewer_{n}.csv" for n in range(1, config["reviewers"] + 1)}
    outputs = {
        "manifest": out_dir / "sample_manifest.json",
        "review": out_dir / "review_items.csv",
        "review_sheet": out_dir / "review_sheet.md",
        "categories": out_dir / "categories.csv",
        **{f"template_{n}": path for n, path in templates.items()},
        "key": out_dir / "blinding_key.csv",
        "report_json": report_dir / "qualitative_sample.json",
        "report_md": report_dir / "qualitative_sample.md",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1
    labelled = {n: p for n, p in templates.items() if p.is_file() and has_annotations(read_table(p))}
    if labelled and not args.keep_labelled:
        logger.error("Annotation templates already hold labels and are never overwritten (see --keep-labelled): %s",
                     [relative(p) for p in labelled.values()])  # fmt: skip
        return 1
    problems = [f"manifests: {e}" for e in verify_checksums(MANIFEST_DIR / "SHA256SUMS")]
    problems += [f"captions: {e}" for e in verify_checksums(OUTPUT_DIR / "captions" / "SHA256SUMS")]
    for problem in problems:
        logger.error("%s", problem)
    if problems:
        return 1

    dataset, models = load_config("dataset", args.overrides), load_config("models", args.overrides)["models"]
    manifest = read_jsonl(resolve_path(dataset["evaluation"]["manifest"]))
    population = [record["sample_id"] for record in manifest]
    references_path = resolve_path(dataset["evaluation"]["references"])
    references = load_references(references_path)
    captions: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in read_jsonl(CAPTIONS):
        captions[record["model_key"]][record["sample_id"]] = record
    if list(captions) != list(models):
        logger.error("Caption models %s do not match models.yaml %s", list(captions), list(models))
        return 1

    sample = select_sample(population, num_recordings, seed)
    key = blind(sample, list(models), seed)
    review, full_key = review_rows(key, references, captions)
    terms = [*models, *(model["display_name"] for model in models.values())]
    file_names = [record["file_name"] for record in manifest if record["sample_id"] in set(sample)]
    if leaks := blinding_problems(review, terms, file_names):
        for leak in leaks:
            logger.error("Blinding: %s", leak)
        return 1
    logger.info("%d recordings, %d review items, %d reviewer templates", len(sample), len(review), len(templates))

    for path in labelled.values():  # a kept template must describe exactly this review
        rows = read_table(path)
        if [[row[c] for c in REVIEW_COLUMNS] for row in rows] != [[row[c] for c in REVIEW_COLUMNS] for row in review]:
            logger.error("%s holds labels for a different review; nothing written", relative(path))
            return 1
        logger.info("Kept the labelled template %s (review columns match)", relative(path))

    write_csv(review, outputs["review"], REVIEW_COLUMNS)
    outputs["review_sheet"].write_text(review_sheet_markdown(review), encoding="utf-8")
    for n, path in templates.items():
        if n not in labelled:
            write_csv(annotation_template(review), path, ANNOTATION_COLUMNS)
    write_csv(config["categories"], outputs["categories"], CATEGORY_COLUMNS)
    write_csv(full_key, outputs["key"], KEY_COLUMNS)
    sample_manifest = {
        "task": "E17",
        "seed": seed,
        "population": len(population),
        "num_recordings": len(sample),
        "num_items": len(review),
        "sample_ids": sample,  # sorted; the code-to-recording map is part of the key
        "reviewers": config["reviewers"],
        "templates": [relative(path) for path in templates.values()],
        "key": {"path": relative(outputs["key"]), "sha256": sha256sum(outputs["key"])},
        "inputs": {
            "captions": {"path": relative(CAPTIONS), "sha256": sha256sum(CAPTIONS)},
            "references": {"path": relative(references_path), "sha256": sha256sum(references_path)},
        },
    }
    write_json(sample_manifest, outputs["manifest"])

    durations = np.array([record["duration_seconds"] for record in manifest])
    values = {
        "Recording": {"duration (s)": durations},
        "Human": {"words (mean of 5)": np.array([np.mean([word_count(r) for r in references[s]]) for s in population])},
    }
    score_sets = {s.system: s for s in read_score_sets(*SCORES)}
    for model_key, model in models.items():
        words = np.array([word_count(captions[model_key][sid]["caption"]) for sid in population])
        if score_sets[model_key].sample_ids != population:
            logger.error("%s: five-reference scores do not cover the manifest in order", model_key)
            return 1
        values[model["display_name"]] = {"words": words} | {
            metric: score_sets[model_key].scores[metric].items for metric in SCORE_METRICS
        }
    report = {
        "run_info": run_info,
        "task": "E17",
        "models": list(models),
        "manifest": sample_manifest,
        "normalization": dict(Counter(row["normalization"] for row in full_key)),
        "comparison": sample_comparison(sample, population, values),
        "limitations": LIMITATIONS,
        "outputs": {name: relative(path) for name, path in outputs.items()},
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    logger.info("Blinding key sha256 %s", sample_manifest["key"]["sha256"])
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
