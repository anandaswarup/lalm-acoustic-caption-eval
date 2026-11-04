"""Score model captions against human references (five- and four-reference conditions).

E12 (`condition: five_reference` in configs/metrics.yaml): every model caption against all five references, with audio,
on every evaluation recording; malformed captions are scored as returned. Writes per-recording and corpus scores,
bootstrap CIs, paired differences between models and the sensitivity check (the recordings where every model's caption
is valid) to outputs/scores/, and outputs/reports/<condition>_evaluation.{json,md}. CLAP audio embeddings are cached in
outputs/cache/ (git-ignored) for E13-E14. four_reference (E14) is not implemented yet. Run with .venv-metrics
(make evaluate).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import sha256sum  # noqa: E402
from src.data.manifest import write_json  # noqa: E402
from src.evaluation.pipeline import add_run_arguments, output_dirs, relative  # noqa: E402
from src.metrics.tools import activate, check_java, java_path  # noqa: E402
from src.utils.cli import base_parser, init_run, not_implemented  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

TASKS = {"five_reference": "E12"}
CAPTIONS = OUTPUT_DIR / "captions" / "all_model_captions.jsonl"
CAPTION_REPORT = OUTPUT_DIR / "reports" / "caption_validation.json"
SMOKE_DIR = OUTPUT_DIR / "smoke" / "evaluation"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    add_run_arguments(parser, SMOKE_DIR)
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("metrics", args.overrides)
    condition = config["condition"]
    if condition == "four_reference":
        return not_implemented(logger, "E14")
    if condition not in TASKS:
        logger.error("Unknown condition %r (expected five_reference or four_reference)", condition)
        return 1
    tools, settings, bootstrap = config["tools"], config["settings"], config["bootstrap"]
    paths = activate(tools, offline=True)  # before anything imports huggingface_hub

    from src.evaluation import references as conditions
    from src.evaluation.inputs import load_model_captions
    from src.evaluation.pipeline import (
        AudioCache,
        clap_cache_key,
        evaluation_inputs,
        existing_outputs,
        timing_block,
        verify_inputs,
    )
    from src.evaluation.scoring import CORPUS_COLUMNS, SCORE_COLUMNS, Evaluator, corpus_rows, score_rows, write_csv
    from src.evaluation.summary import (
        ALL_RECORDINGS,
        DIFFERENCE_COLUMNS,
        SENSITIVITY_COLUMNS,
        SUMMARY_COLUMNS,
        VALID_FOR_ALL,
        difference_rows,
        report_markdown,
        restrict,
        sensitivity_rows,
        summary_rows,
    )
    from src.evaluation.validation import LIMITATIONS, package_versions
    from src.metrics.registry import check_scores

    scores_dir, report_dir = output_dirs(args, SMOKE_DIR)
    outputs = {
        "scores": scores_dir / f"{condition}_scores.csv",
        "corpus": scores_dir / f"{condition}_corpus.csv",
        "summary": scores_dir / f"{condition}_summary.csv",
        "differences": scores_dir / f"{condition}_differences.csv",
        "sensitivity": scores_dir / f"{condition}_sensitivity.csv",
        "report_json": report_dir / f"{condition}_evaluation.json",
        "report_md": report_dir / f"{condition}_evaluation.md",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1

    if problems := verify_inputs(paths["root"]):
        for problem in problems:
            logger.error("%s", problem)
        return 1
    java_version = check_java(java_path(tools))

    dataset, models = load_config("dataset", args.overrides), load_config("models", args.overrides)["models"]
    captions = load_model_captions(CAPTIONS)
    if set(captions) != set(models):
        logger.error("Caption models %s do not match models.yaml %s", sorted(captions), sorted(models))
        return 1
    caption_report = json.loads(CAPTION_REPORT.read_text(encoding="utf-8"))
    if caption_report["output"]["sha256"] != sha256sum(CAPTIONS):
        logger.error("%s describes a different caption file", relative(CAPTION_REPORT))
        return 1
    valid_ids = caption_report["valid_for_all_models"]["sample_ids"]
    manifest, references_path, references, audio = evaluation_inputs(dataset, args.limit)
    if args.limit:
        logger.warning("Smoke test: first %d recordings only", len(manifest))
    sample_ids = [record["sample_id"] for record in manifest]
    valid_ids = [sid for sid in valid_ids if sid in set(sample_ids)]
    logger.info("%d recordings x %d models; %d with every caption valid", len(sample_ids), len(models), len(valid_ids))

    evaluator = Evaluator(settings, tools)
    evaluator.load_models(audio=True)
    cache = AudioCache(None if args.cache == "none" else resolve_path(args.cache), clap_cache_key(settings, tools))
    hits = cache.preload(evaluator.clap, manifest, audio)
    logger.info("CLAP audio embeddings from the cache: %d of %d", hits, len(manifest))

    score_sets = []
    for model in models:
        logger.info("Scoring %s", model)
        candidates = {sid: captions[model][sid] for sid in sample_ids}
        score_set = evaluator.score(model, condition, conditions.ALL, sample_ids, candidates, references, audio)
        if errors := check_scores(score_set.scores, len(sample_ids)):
            logger.error("%s: invalid scores: %s", model, errors)
            return 1
        score_sets.append(score_set)
    if cache.path:
        total = cache.save(evaluator.clap, manifest, audio)
        logger.info("CLAP audio cache: %d recordings in %s", total, relative(cache.path))

    boot = (bootstrap["num_resamples"], bootstrap["confidence"], bootstrap["seed"])
    valid_sets = restrict(score_sets, valid_ids)
    summary = summary_rows(score_sets, ALL_RECORDINGS, *boot) + summary_rows(valid_sets, VALID_FOR_ALL, *boot)
    differences = difference_rows(score_sets, ALL_RECORDINGS, *boot) + difference_rows(valid_sets, VALID_FOR_ALL, *boot)
    full = [r for r in summary if r["sample"] == ALL_RECORDINGS]
    sensitivity = sensitivity_rows(
        full,
        [r for r in summary if r["sample"] == VALID_FOR_ALL],
        [r for r in differences if r["sample"] == ALL_RECORDINGS],
        [r for r in differences if r["sample"] == VALID_FOR_ALL],
    )

    write_csv([row for s in score_sets for row in score_rows(s)], outputs["scores"], SCORE_COLUMNS)
    write_csv([row for s in score_sets for row in corpus_rows(s)], outputs["corpus"], CORPUS_COLUMNS)
    write_csv(summary, outputs["summary"], SUMMARY_COLUMNS)
    write_csv(differences, outputs["differences"], DIFFERENCE_COLUMNS)
    write_csv(sensitivity, outputs["sensitivity"], SENSITIVITY_COLUMNS)

    per_model = caption_report["per_model"]
    report = {
        "run_info": run_info,
        "task": TASKS[condition],
        "condition": condition,
        "subset": conditions.ALL,
        "smoke_test_limit": args.limit,
        "display_names": {key: model["display_name"] for key, model in models.items()},
        "metrics": list(score_sets[0].scores),
        "inputs": {
            "recordings": len(sample_ids),
            "references_per_recording": len(references[sample_ids[0]]),
            "captions": {"path": relative(CAPTIONS), "sha256": sha256sum(CAPTIONS)},
            "references": {"path": relative(references_path), "sha256": sha256sum(references_path)},
            "normalization": {key: per_model[key]["normalization"] for key in models},
        },
        "bootstrap": bootstrap,
        "settings": settings,
        "java_version": java_version,
        "packages": package_versions(),
        "summary": full + [r for r in summary if r["sample"] == VALID_FOR_ALL],
        "differences": differences,
        "sensitivity": {"num_items": len(valid_ids), "excluded": len(sample_ids) - len(valid_ids), "rows": sensitivity},
        "timing": timing_block(evaluator.timer, hits),
        "outputs": {name: relative(path) for name, path in outputs.items()},
        "limitations": LIMITATIONS,
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
