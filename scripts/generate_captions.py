"""Generate one caption per recording with a selected LALM (E04-E08).

Resumable: samples that already have an `ok` record in the output JSONL are skipped, so an interrupted run is
continued by running the same command again. Every attempt is appended; the last record per sample_id counts.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import sha256sum  # noqa: E402
from src.generation.runner import load_samples, run_generation  # noqa: E402
from src.models.base import load_adapter, merge_decoding  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--run", help="run from generation.yaml `runs` (default: generation.yaml `run`)")
    parser.add_argument("--limit", type=int, help="caption only the first N samples of the manifest")
    parser.add_argument("--sample-ids", nargs="+", help="caption only these sample_ids")
    parser.add_argument("--output-dir", help="override the run's output_dir")
    parser.add_argument("--no-resume", action="store_true", help="caption every sample, even those already ok")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)

    generation = load_config("generation", args.overrides)
    models = load_config("models", args.overrides)["models"]
    dataset = load_config("dataset")
    run = args.run or generation["run"]
    if run not in generation["runs"]:
        hint = " (consolidation is E09: scripts/consolidate_captions.py)" if run == "consolidate" else ""
        logger.error("Unknown run %r; expected one of %s%s", run, sorted(generation["runs"]), hint)
        return 1
    if args.model not in models:
        logger.error("Unknown model %r; expected one of %s", args.model, sorted(models))
        return 1
    model = models[args.model]
    if (key_env := model.get("api_key_env")) and model.get("adapter") != "mock" and not os.environ.get(key_env):
        logger.error("%s is not set (add it to .env)", key_env)
        return 1

    run_config = generation["runs"][run]
    manifest = resolve_path(run_config["manifest"])
    output = resolve_path(args.output_dir or run_config["output_dir"]) / model["output"]
    clotho_root = resolve_path(dataset["root"])
    samples = load_samples(manifest, clotho_root, args.limit, args.sample_ids)
    decoding = merge_decoding(generation, model)
    requests_config = {**generation["requests"], **(model.get("requests_overrides") or {})}
    if not generation.get("frozen"):
        logger.warning("generation.yaml is not frozen yet (it is frozen before caption generation)")

    meta = {
        "run_info": collect_run_info(Path(__file__).stem, seed=args.seed),
        "run": run,
        "model_key": args.model,
        "model": model,
        "manifest": {"path": str(run_config["manifest"]), "sha256": sha256sum(manifest)},
        "selection": {"limit": args.limit, "sample_ids": args.sample_ids, "resume": not args.no_resume},
        "prompt": generation["prompt"],
        "decoding": decoding,
        "requests": requests_config,
        "audio": generation["audio"],
        "generation_frozen": generation.get("frozen", False),
    }
    adapter = load_adapter(args.model, model, generation)
    summary = run_generation(
        adapter,
        samples,
        output,
        generation["prompt"]["text"],
        decoding,
        requests_config,
        meta,
        resume=generation.get("resume", True) and not args.no_resume,
    )
    logger.info("Summary: %s", {key: summary[key] for key in ("samples", "with_record", "statuses", "latency_s")})
    if summary["failed"]:
        logger.warning("%d samples not ok: %s", len(summary["failed"]), summary["failed"][:10])
    errors = summary["statuses"].get("error", 0) + len(summary["missing"])
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
