"""Validate the caption metrics and write the metric implementation report (E11).

Runs the metric integration tests, then scores the committed pilot captions (20 validation-split recordings, so no
evaluation-split scores exist before E12) under every condition: five_reference, four_reference drop_1-5 for the four
models and human_loo drop_1-5, all with audio. The whole scoring is run twice with freshly loaded models and must be
bit-identical. Per-step timings are extrapolated to the full evaluation set, and inputs longer than a model's limit are
counted on the frozen evaluation captions and references. Writes outputs/reports/metric_implementation.{json,md} and
the pilot score tables under outputs/scores/pilot/. Run with .venv-metrics (make validate-metrics).
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import read_jsonl, sha256sum  # noqa: E402
from src.data.manifest import verify_checksums, write_json  # noqa: E402
from src.metrics.tools import activate, check_java, java_path, verify_manifest  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import MANIFEST_DIR, OUTPUT_DIR, REPO_ROOT, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

HUMAN = "human"


def run_integration_tests(logger) -> dict:
    logger.info("Running the metric integration tests")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "metrics", "tests/test_metrics_integration.py", "-q"]
        + ["-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    summary = next(
        (line.strip("= ") for line in reversed(result.stdout.splitlines()) if re.search(r"\d+ (passed|failed)", line)),
        "no summary",
    )
    logger.info("Integration tests: %s", summary)
    return {"summary": summary, "returncode": result.returncode}


def pilot_inputs(generation: dict, models: dict, dataset: dict) -> tuple[list[dict], dict, dict, dict]:
    """Pilot manifest, references, model captions (normalized with the E10 rule) and normalization counts."""
    from src.generation.consolidate import consolidate

    pilot = dataset["pilot"]
    manifest_path = resolve_path(pilot["manifest"])
    manifest = read_jsonl(manifest_path)
    references = {record["sample_id"]: record["captions"] for record in manifest}
    source_dir = resolve_path(generation["runs"]["pilot"]["output_dir"])
    sources = {key: source_dir / model["output"] for key, model in models.items()}
    with tempfile.TemporaryDirectory() as tmp:
        refs_path = Path(tmp) / "pilot_references.json"
        refs_path.write_text(json.dumps(references), encoding="utf-8")
        records, report, errors = consolidate(
            sources, manifest_path, refs_path, generation["prompt"]["text"], len(manifest[0]["captions"])
        )
    if errors:
        raise ValueError(f"Pilot captions are not complete: {errors[:5]}")
    captions: dict[str, dict[str, str]] = {key: {} for key in sources}
    normalization: dict[str, dict[str, int]] = {key: {} for key in sources}
    for record in records:
        captions[record["model_key"]][record["sample_id"]] = record["caption"]
        counts = normalization[record["model_key"]]
        counts[record["normalization"]] = counts.get(record["normalization"], 0) + 1
    return manifest, references, captions, normalization


def score_all(evaluator, manifest, references, captions, audio) -> list:
    from src.evaluation import references as conditions

    sample_ids = [record["sample_id"] for record in manifest]
    sets = [
        evaluator.score(model, conditions.FIVE_REFERENCE, conditions.ALL, sample_ids, caps, references, audio)
        for model, caps in captions.items()
    ]
    for subset in conditions.subsets(conditions.FOUR_REFERENCE):
        if errors := conditions.check_split(references, subset):
            raise ValueError(f"{subset}: {errors[:5]}")
        refs = conditions.subset_references(references, subset)
        for model, caps in captions.items():
            sets.append(evaluator.score(model, conditions.FOUR_REFERENCE, subset, sample_ids, caps, refs, audio))
        held_out = conditions.held_out_captions(references, subset)
        sets.append(evaluator.score(HUMAN, conditions.HUMAN_LOO, subset, sample_ids, held_out, refs, audio))
    return sets


def scaling_call(evaluator, manifest, references, captions, audio) -> tuple[dict[str, float], int]:
    """Seconds per step for one five_reference call on every pilot model caption at once (for the timing estimate)."""
    from src.evaluation import references as conditions
    from src.metrics.registry import Timer

    evaluator.timer = evaluator.reference.timer = Timer()
    ids = [f"{model}:{record['sample_id']}" for model in captions for record in manifest]
    candidates = {key: captions[key.split(":")[0]][key.split(":")[1]] for key in ids}
    refs = {key: references[key.split(":")[1]] for key in ids}
    sounds = {key: audio[key.split(":")[1]] for key in ids}
    evaluator.score("all_models", conditions.FIVE_REFERENCE, conditions.ALL, ids, candidates, refs, sounds)
    return dict(evaluator.timer.seconds), len(ids)


def ptb_groups(evaluator, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    from src.metrics.preprocess import ptb_tokenize

    flat = [text for texts in groups.values() for text in texts]
    root = evaluator.paths["root"]
    tokenized = iter(ptb_tokenize(flat, java=evaluator.java, cache_path=root, tmp_path=root / "tmp"))
    return {name: [next(tokenized) for _ in texts] for name, texts in groups.items()}


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--report-dir", default=str(OUTPUT_DIR / "reports"), help="default: %(default)s")
    parser.add_argument("--scores-dir", default=str(OUTPUT_DIR / "scores" / "pilot"), help="default: %(default)s")
    parser.add_argument("--runs", type=int, default=2, help="scoring repetitions to compare (default: %(default)s)")
    parser.add_argument("--skip-tests", action="store_true", help="do not run the integration tests")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    metrics_config = load_config("metrics", args.overrides)
    tools, settings = metrics_config["tools"], metrics_config["settings"]
    paths = activate(tools, offline=True)  # before anything imports huggingface_hub

    from src.evaluation.inputs import audio_paths, load_model_captions, load_references
    from src.evaluation.scoring import CORPUS_COLUMNS, SCORE_COLUMNS, Evaluator, corpus_rows, score_rows, write_csv
    from src.evaluation.validation import (
        LIMITATIONS,
        compare_runs,
        corpus_table,
        extrapolate,
        package_versions,
        report_markdown,
        truncation_counts,
        two_point_estimate,
    )
    from src.metrics.registry import METRICS

    problems = verify_manifest(paths["root"])
    problems += [f"manifests: {e}" for e in verify_checksums(MANIFEST_DIR / "SHA256SUMS")]
    problems += [f"captions: {e}" for e in verify_checksums(OUTPUT_DIR / "captions" / "SHA256SUMS")]
    if problems:
        for problem in problems:
            logger.error("%s", problem)
        return 1
    java_version = check_java(java_path(tools))

    tests = {"summary": "skipped", "returncode": None} if args.skip_tests else run_integration_tests(logger)
    generation, models = load_config("generation", args.overrides), load_config("models", args.overrides)["models"]
    dataset = load_config("dataset", args.overrides)
    manifest, references, captions, normalization = pilot_inputs(generation, models, dataset)
    audio = audio_paths(manifest, resolve_path(dataset["root"]))
    logger.info("Pilot: %d recordings, models %s; normalization %s", len(manifest), list(captions), normalization)

    runs, timers = [], []
    for run in range(1, args.runs + 1):
        evaluator = Evaluator(settings, tools)
        evaluator.load_models(audio=True)
        logger.info("Run %d: scoring %d calls", run, 5 * (len(captions) + 1) + len(captions))
        runs.append(score_all(evaluator, manifest, references, captions, audio))
        timers.append(evaluator.timer)
        seconds = sum(value for name, value in evaluator.timer.seconds.items() if name != "load_models")
        logger.info("Run %d done: %.0f s of scoring", run, seconds)
    determinism = {"runs": args.runs, **compare_runs(runs[0], runs[-1])} if args.runs > 1 else {"runs": 1}
    logger.info("Timing one call on all pilot model captions")
    large, large_items = scaling_call(evaluator, manifest, references, captions, audio)
    logger.info("Bit-identical across runs: %s", determinism.get("identical"))

    scores_dir = resolve_path(args.scores_dir)
    scores_path = scores_dir / "validation_scores.csv"
    write_csv([row for s in runs[0] for row in score_rows(s)], scores_path, SCORE_COLUMNS)
    write_csv([row for s in runs[0] for row in corpus_rows(s)], scores_dir / "validation_corpus.csv", CORPUS_COLUMNS)

    logger.info("Measuring input lengths on the frozen evaluation captions and references")
    evaluation = dataset["evaluation"]
    eval_captions = load_model_captions(OUTPUT_DIR / "captions" / "all_model_captions.jsonl")
    eval_refs = load_references(resolve_path(evaluation["references"]))
    groups = {key: list(caps.values()) for key, caps in eval_captions.items()}
    groups["references"] = [ref for refs in eval_refs.values() for ref in refs]
    groups = ptb_groups(evaluator, groups)
    sbert_limit = evaluator.reference._model("sbert").max_seq_length
    checks = [
        {"input": "BERTScore (roberta-large)", "tokenizer": settings["bertscore"]["model"],
         "limit": settings["bertscore"]["max_length"]},
        {"input": "Fluency-error detector (bert-base-uncased)", "tokenizer": "bert-base-uncased",
         "limit": settings["fluency_error"]["max_length"]},
        {"input": "SBERT (paraphrase-TinyBERT-L6-v2)", "tokenizer": "sentence-transformers/paraphrase-TinyBERT-L6-v2",
         "limit": sbert_limit},
        {"input": "CLAP text (gpt2)", "tokenizer": "gpt2", "suffix": " <|endoftext|>",
         "limit": evaluator.clap.model.args.text_len},
    ]  # fmt: skip
    truncation = truncation_counts(groups, checks)

    timer = timers[0]
    pins = {repo: revision for repo, revision in tools["hf_models"].items()}
    pins |= {name: sha for name, sha in tools["files"].items()}
    pins |= {f"{tools['clap']['repo']}/{tools['clap']['filename']}": tools["clap"]["sha256"]}
    pins |= {tools["bertscore_baseline"]["path"]: tools["bertscore_baseline"]["sha256"]}
    manifest_json = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    passed = determinism.get("identical", True) and tests["returncode"] in (0, None)
    report = {
        "run_info": run_info,
        "passed": passed,
        "tools": {
            "platform": manifest_json["platform"],
            "java_version": java_version,
            "jdk_release": manifest_json["jdk_release"],
            "manifest_files": len(manifest_json["files"]),
            "manifest_sha256": sha256sum(paths["manifest"]),
            "manifest_ok": True,
            "pins": pins,
        },
        "packages": package_versions(),
        "preprocessing": "PTB tokenization (aac-metrics, lowercased, punctuation removed) for every text metric; "
        "audio read at 44.1 kHz mono with no resampling.",
        "settings": settings,
        "metrics": [
            {
                "name": m.name,
                "label": m.label,
                "family": m.family,
                "corpus_is_item_mean": m.corpus_is_item_mean,
                "needs_audio": m.needs_audio,
            }
            for m in METRICS
        ],
        "integration_tests": tests,
        "pilot": {
            "recordings": len(manifest),
            "systems": [*captions, HUMAN],
            "normalization": normalization,
            "calls_per_run": len(runs[0]),
            "five_reference": corpus_table(runs[0], "five_reference"),
            "scores": str(scores_path.relative_to(REPO_ROOT) if scores_path.is_relative_to(REPO_ROOT) else scores_path),
        },
        "determinism": determinism,
        "truncation": truncation,
        "timing": {
            "seconds": {name: round(value, 2) for name, value in sorted(timer.seconds.items())},
            "calls": dict(timer.calls),
            "load_models_seconds": round(timer.seconds["load_models"], 1),
            "extrapolation": extrapolate(dict(timer.seconds), len(manifest), evaluation["expected_recordings"]),
            "scaling": {
                "large_items": large_items,
                "large_seconds": {name: round(value, 2) for name, value in sorted(large.items())},
                "estimate": two_point_estimate(
                    {name: timer.seconds[name] / timer.calls[name] for name in timer.seconds},
                    len(manifest),
                    large,
                    large_items,
                    evaluation["expected_recordings"],
                    len(runs[0]),
                ),
            },
        },
        "limitations": LIMITATIONS,
    }
    report_dir = resolve_path(args.report_dir)
    write_json(report, report_dir / "metric_implementation.json")
    (report_dir / "metric_implementation.md").write_text(report_markdown(report), encoding="utf-8")
    logger.info("Wrote metric_implementation.{json,md} to %s; passed: %s", report_dir, passed)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
