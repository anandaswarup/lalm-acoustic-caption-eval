"""Human leave-one-out calibration: score each held-out caption against the other four.

E12: for k = 1..5, the human caption at CSV position k of every evaluation recording is scored against the other four
(subset drop_k, shared with E13), with audio. The checks that each split is valid run before scoring. Writes the
per-recording, corpus and summary scores (per position and averaged over the five) to outputs/scores/human_loo_*.csv
and outputs/reports/human_loo_evaluation.{json,md}. Near-paraphrase sensitivity: each held-out caption's word-set
Jaccard with its closest reference is saved (human_loo_overlap.csv, for E13), and the averaged baseline is also
summarized on the held-out captions below each `leave_one_out.overlap_thresholds` value, without rescoring. CLAP audio
embeddings come from outputs/cache/ (git-ignored). Run with .venv-metrics (make loo).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import sha256sum  # noqa: E402
from src.data.manifest import write_json  # noqa: E402
from src.evaluation.pipeline import add_run_arguments, output_dirs, relative  # noqa: E402
from src.metrics.tools import activate, check_java, java_path  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

HUMAN = "human"
SMOKE_DIR = OUTPUT_DIR / "smoke" / "loo"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    add_run_arguments(parser, SMOKE_DIR)
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("metrics", args.overrides)
    tools, settings, bootstrap, loo = config["tools"], config["settings"], config["bootstrap"], config["leave_one_out"]
    paths = activate(tools, offline=True)  # before anything imports huggingface_hub

    from src.evaluation import references as conditions
    from src.evaluation.calibration import (
        COMPARABILITY,
        OVERLAP_COLUMNS,
        calibration_checks,
        closest_reference_overlap,
        duplicate_held_out,
        overlap_keep_masks,
        overlap_rows,
        overlap_sample,
        overlap_summary,
        report_markdown,
    )
    from src.evaluation.metric_info import LIMITATIONS, package_versions
    from src.evaluation.pipeline import (
        AudioCache,
        clap_cache_key,
        evaluation_inputs,
        existing_outputs,
        timing_block,
        verify_inputs,
    )
    from src.evaluation.scoring import CORPUS_COLUMNS, SCORE_COLUMNS, Evaluator, corpus_rows, score_rows, write_csv
    from src.evaluation.summary import ALL_RECORDINGS, SUMMARY_COLUMNS, average_subsets, masked_average, summary_rows
    from src.metrics.preprocess import ptb_tokenize
    from src.metrics.registry import check_scores

    condition = conditions.HUMAN_LOO
    scores_dir, report_dir = output_dirs(args, SMOKE_DIR)
    outputs = {
        "scores": scores_dir / f"{condition}_scores.csv",
        "corpus": scores_dir / f"{condition}_corpus.csv",
        "summary": scores_dir / f"{condition}_summary.csv",
        "overlap": scores_dir / f"{condition}_overlap.csv",
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

    dataset = load_config("dataset", args.overrides)
    manifest, references_path, references, audio = evaluation_inputs(dataset, args.limit)
    if args.limit:
        logger.warning("Smoke test: first %d recordings only", len(manifest))
    sample_ids = [record["sample_id"] for record in manifest]
    subsets = conditions.subsets(condition, loo["num_positions"])

    checks = calibration_checks(references, loo["num_positions"])
    for check in checks:
        logger.info("Check %s: %s", "passed" if check["passed"] else "FAILED", check["name"])
    if not all(check["passed"] for check in checks):
        logger.error("Calibration checks failed; nothing scored")
        return 1

    evaluator = Evaluator(settings, tools)
    evaluator.load_models(audio=True)
    root = evaluator.paths["root"]
    flat = [ref for sid in sample_ids for ref in references[sid]]
    tokenized = iter(ptb_tokenize(flat, java=evaluator.java, cache_path=root, tmp_path=root / "tmp"))
    tokenized = {sid: [next(tokenized) for _ in references[sid]] for sid in sample_ids}
    duplicates = duplicate_held_out(tokenized)
    logger.info("Held-out captions identical to one of their references after PTB: %d", len(duplicates))
    thresholds = loo["overlap_thresholds"]
    overlaps = closest_reference_overlap(tokenized)
    overlap = overlap_summary(overlaps, thresholds)
    for row in overlap["thresholds"]:
        logger.info("Closest-reference Jaccard >= %s: %d held-out captions", row["threshold"], row["at_or_above"])
    cache = AudioCache(None if args.cache == "none" else resolve_path(args.cache), clap_cache_key(settings, tools))
    hits = cache.preload(evaluator.clap, manifest, audio)
    logger.info("CLAP audio embeddings from the cache: %d of %d", hits, len(manifest))

    score_sets = []
    for subset in subsets:
        refs = conditions.subset_references(references, subset)
        if wrong := [sid for sid in sample_ids if len(refs[sid]) != loo["references_per_subset"]]:
            logger.error("%s: %d recordings without %d references", subset, len(wrong), loo["references_per_subset"])
            return 1
        logger.info("Scoring %s %s", HUMAN, subset)
        held_out = conditions.held_out_captions(references, subset)
        score_set = evaluator.score(HUMAN, condition, subset, sample_ids, held_out, refs, audio)
        if errors := check_scores(score_set.scores, len(sample_ids)):
            logger.error("%s: invalid scores: %s", subset, errors)
            return 1
        score_sets.append(score_set)
    if cache.path:
        total = cache.save(evaluator.clap, manifest, audio)
        logger.info("CLAP audio cache: %d recordings in %s", total, relative(cache.path))

    averaged = average_subsets(score_sets)
    boot = (bootstrap["num_resamples"], bootstrap["confidence"], bootstrap["seed"])
    summary = [row for s in [*score_sets, averaged] for row in summary_rows([s], ALL_RECORDINGS, *boot)]
    for threshold in thresholds:
        kept = masked_average(score_sets, overlap_keep_masks(overlaps, sample_ids, subsets, threshold))
        summary += summary_rows([kept], overlap_sample(threshold), *boot)
    write_csv([row for s in score_sets for row in score_rows(s)], outputs["scores"], SCORE_COLUMNS)
    write_csv([row for s in [*score_sets, averaged] for row in corpus_rows(s)], outputs["corpus"], CORPUS_COLUMNS)
    write_csv(summary, outputs["summary"], SUMMARY_COLUMNS)
    write_csv(overlap_rows(overlaps), outputs["overlap"], OVERLAP_COLUMNS)

    report = {
        "run_info": run_info,
        "task": "E12",
        "condition": condition,
        "subsets": subsets,
        "smoke_test_limit": args.limit,
        "metrics": list(averaged.scores),
        "inputs": {
            "recordings": len(sample_ids),
            "held_out_captions": len(sample_ids) * len(subsets),
            "references_per_subset": loo["references_per_subset"],
            "references": {"path": relative(references_path), "sha256": sha256sum(references_path)},
        },
        "checks": checks,
        "duplicates": duplicates,
        "overlap": overlap,
        "bootstrap": bootstrap,
        "settings": settings,
        "java_version": java_version,
        "packages": package_versions(),
        "summary": summary,
        "timing": timing_block(evaluator.timer, hits),
        "outputs": {name: relative(path) for name, path in outputs.items()},
        "limitations": COMPARABILITY + [item for item in LIMITATIONS if not item.startswith("Malformed")],
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
