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
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import read_jsonl, sha256sum  # noqa: E402
from src.data.manifest import verify_checksums, write_json  # noqa: E402
from src.metrics.tools import activate, check_java, java_path, verify_manifest  # noqa: E402
from src.utils.cli import base_parser, init_run, not_implemented  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import MANIFEST_DIR, OUTPUT_DIR, REPO_ROOT, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

TASKS = {"five_reference": "E12"}
CAPTIONS = OUTPUT_DIR / "captions" / "all_model_captions.jsonl"
CAPTION_REPORT = OUTPUT_DIR / "reports" / "caption_validation.json"
SMOKE_DIR = OUTPUT_DIR / "smoke" / "evaluation"


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)


def clap_cache_key(settings: dict, tools: dict) -> str:
    """What the cached audio embeddings depend on: CLAP settings, weights and msclap version."""
    key = {"clap": settings["clap"], "weights_sha256": tools["clap"]["sha256"], "msclap": metadata.version("msclap")}
    return json.dumps(key, sort_keys=True)


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    smoke = f"{relative(SMOKE_DIR)} with --limit"
    parser.add_argument("--scores-dir", help=f"default: {relative(OUTPUT_DIR / 'scores')} ({smoke})")
    parser.add_argument("--report-dir", help=f"default: {relative(OUTPUT_DIR / 'reports')} ({smoke})")
    cache = relative(OUTPUT_DIR / "cache" / "clap_audio.npz")
    parser.add_argument("--cache", default=cache, help="CLAP audio cache, 'none' to disable (default: %(default)s)")
    parser.add_argument("--limit", type=int, help="score the first N recordings only (smoke test)")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
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
    from src.evaluation.inputs import audio_paths, load_model_captions, load_references
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
    from src.metrics.clap import load_audio_cache, save_audio_cache
    from src.metrics.registry import check_scores

    out_default = SMOKE_DIR if args.limit else None
    scores_dir = resolve_path(args.scores_dir or out_default or OUTPUT_DIR / "scores")
    report_dir = resolve_path(args.report_dir or out_default or OUTPUT_DIR / "reports")
    outputs = {
        "scores": scores_dir / f"{condition}_scores.csv",
        "corpus": scores_dir / f"{condition}_corpus.csv",
        "summary": scores_dir / f"{condition}_summary.csv",
        "differences": scores_dir / f"{condition}_differences.csv",
        "sensitivity": scores_dir / f"{condition}_sensitivity.csv",
        "report_json": report_dir / f"{condition}_evaluation.json",
        "report_md": report_dir / f"{condition}_evaluation.md",
    }
    if not args.force and (existing := [relative(p) for p in outputs.values() if p.exists()]):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1

    problems = verify_manifest(paths["root"])
    problems += [f"manifests: {e}" for e in verify_checksums(MANIFEST_DIR / "SHA256SUMS")]
    problems += [f"captions: {e}" for e in verify_checksums(OUTPUT_DIR / "captions" / "SHA256SUMS")]
    if problems:
        for problem in problems:
            logger.error("%s", problem)
        return 1
    java_version = check_java(java_path(tools))

    dataset, models = load_config("dataset", args.overrides), load_config("models", args.overrides)["models"]
    evaluation = dataset["evaluation"]
    manifest = read_jsonl(resolve_path(evaluation["manifest"]))
    references_path = resolve_path(evaluation["references"])
    references = load_references(references_path)
    captions = load_model_captions(CAPTIONS)
    if set(captions) != set(models):
        logger.error("Caption models %s do not match models.yaml %s", sorted(captions), sorted(models))
        return 1
    caption_report = json.loads(CAPTION_REPORT.read_text(encoding="utf-8"))
    if caption_report["output"]["sha256"] != sha256sum(CAPTIONS):
        logger.error("%s describes a different caption file", relative(CAPTION_REPORT))
        return 1
    valid_ids = caption_report["valid_for_all_models"]["sample_ids"]
    if args.limit:
        manifest = manifest[: args.limit]
        logger.warning("Smoke test: first %d recordings only", len(manifest))
    sample_ids = [record["sample_id"] for record in manifest]
    references = {sid: references[sid] for sid in sample_ids}
    valid_ids = [sid for sid in valid_ids if sid in set(sample_ids)]
    audio = audio_paths(manifest, resolve_path(dataset["root"]))
    logger.info("%d recordings x %d models; %d with every caption valid", len(sample_ids), len(models), len(valid_ids))

    evaluator = Evaluator(settings, tools)
    evaluator.load_models(audio=True)
    cache_path = None if args.cache == "none" else resolve_path(args.cache)
    cache_key = clap_cache_key(settings, tools)
    cached = load_audio_cache(cache_path, cache_key) if cache_path else {}
    hits = {str(audio[r["sample_id"]]): cached[r["audio_sha256"]] for r in manifest if r["audio_sha256"] in cached}
    evaluator.clap.preload_audio(hits)
    logger.info("CLAP audio embeddings from the cache: %d of %d", len(hits), len(manifest))

    score_sets = []
    for model in models:
        logger.info("Scoring %s", model)
        candidates = {sid: captions[model][sid] for sid in sample_ids}
        score_set = evaluator.score(model, condition, conditions.ALL, sample_ids, candidates, references, audio)
        if errors := check_scores(score_set.scores, len(sample_ids)):
            logger.error("%s: invalid scores: %s", model, errors)
            return 1
        score_sets.append(score_set)
    if cache_path:
        by_path = evaluator.clap.audio_embeddings()
        computed = {r["audio_sha256"]: by_path[str(audio[r["sample_id"]])] for r in manifest}
        save_audio_cache(cache_path, cache_key, cached | computed)
        logger.info("CLAP audio cache: %d recordings in %s", len(cached | computed), relative(cache_path))

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

    timer = evaluator.timer
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
        "timing": {
            "seconds": {name: round(value, 2) for name, value in timer.seconds.items() if name != "load_models"},
            "calls": dict(timer.calls),
            "load_models_seconds": round(timer.seconds["load_models"], 1),
            "scoring_seconds": round(sum(v for name, v in timer.seconds.items() if name != "load_models"), 1),
            "clap_cache_hits": len(hits),
        },
        "outputs": {name: relative(path) for name, path in outputs.items()},
        "limitations": LIMITATIONS,
    }
    write_json(report, outputs["report_json"])
    outputs["report_md"].write_text(report_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", sorted(report["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
