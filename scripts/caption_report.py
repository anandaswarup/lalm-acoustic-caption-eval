"""Write the completion report for one model's caption run (E06-E09).

Reads the run's captions JSONL (latest record per sample_id) and its meta file, checks them against the run's
manifest and the frozen prompt, and writes `<model>_<run>_completion.{json,md}`. Reports hold counts and
sample_ids only, never caption text. Exits 1 when the run is incomplete.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import write_json  # noqa: E402
from src.generation.report import completion_report, report_markdown  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--run", help="run from generation.yaml `runs` (default: generation.yaml `run`)")
    parser.add_argument("--output-dir", help="override the run's output_dir (where the captions are read from)")
    parser.add_argument("--report-dir", default=str(OUTPUT_DIR / "reports"), help="default: %(default)s")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)

    generation = load_config("generation", args.overrides)
    models = load_config("models", args.overrides)["models"]
    run = args.run or generation["run"]
    if run not in generation["runs"]:
        logger.error("Unknown run %r; expected one of %s", run, sorted(generation["runs"]))
        return 1
    if args.model not in models:
        logger.error("Unknown model %r; expected one of %s", args.model, sorted(models))
        return 1
    run_config = generation["runs"][run]
    output = resolve_path(args.output_dir or run_config["output_dir"]) / models[args.model]["output"]
    if not output.exists():
        logger.error("No captions at %s", output)
        return 1
    meta_path = output.with_name(output.name.removesuffix(".jsonl") + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if meta is None:
        logger.warning("No meta file at %s", meta_path)

    report = completion_report(output, resolve_path(run_config["manifest"]), generation["prompt"]["text"], meta)
    report = {"run_info": collect_run_info(Path(__file__).stem, seed=args.seed), "model_key": args.model, **report}
    report_dir = resolve_path(args.report_dir)
    stem = f"{args.model}_{run}_completion"
    write_json(report, report_dir / f"{stem}.json")
    title = f"Caption completion report: {models[args.model].get('display_name', args.model)} ({run})"
    (report_dir / f"{stem}.md").write_text(report_markdown(report, title), encoding="utf-8")
    logger.info("Wrote %s.{json,md} to %s", stem, report_dir)

    problems, fmt = report["problems"], report["format"]
    logger.info("Statuses: %s; complete: %s", report["statuses"], report["complete"])
    logger.info(
        'Format: %d retried, %d fixed by a retry, %d still JSON-like (%d exactly {"..."})',
        fmt["retried"],
        fmt["fixed_by_retry"],
        len(fmt["json_like"]),
        len(fmt["json_like_unwrappable"]),
    )
    for name, values in problems.items():
        if values:
            logger.warning("%s: %d (%s)", name, len(values), values[:10])
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
