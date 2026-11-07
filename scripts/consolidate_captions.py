"""Consolidate and freeze the four models' evaluation captions (E09).

Checks that every model has one `ok` caption per evaluation recording, produced with the frozen prompt, normalizes
the captions (unwraps exactly {"<text>"}, flags other JSON-like output as malformed, scored as returned) and writes
`all_model_captions.jsonl`, a SHA256SUMS covering it and the source files, and the caption validation report. If any
check fails, the report is written but the dataset is not. Existing frozen outputs are never overwritten without
--force.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import sha256sum  # noqa: E402
from src.data.manifest import verify_checksums, write_checksums, write_json, write_jsonl  # noqa: E402
from src.generation.consolidate import consolidate, report_markdown  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

OUTPUT_NAME = "all_model_captions.jsonl"
CHECKSUM_FILE = "SHA256SUMS"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "captions"), help="default: %(default)s")
    parser.add_argument("--report-dir", default=str(OUTPUT_DIR / "reports"), help="default: %(default)s")
    parser.add_argument("--force", action="store_true", help="overwrite an existing frozen caption dataset")
    parser.add_argument("--check", action="store_true", help="only verify the frozen files against SHA256SUMS")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)

    output_dir = resolve_path(args.output_dir)
    output, checksum_path = output_dir / OUTPUT_NAME, output_dir / CHECKSUM_FILE
    if args.check:
        errors = verify_checksums(checksum_path)
        for error in errors:
            logger.error("check: %s", error)
        if not errors:
            logger.info("All files in %s match", checksum_path)
        return 1 if errors else 0
    if existing := [path for path in (output, checksum_path) if path.exists()]:
        if not args.force:
            logger.error("Frozen captions already exist (%s); use --force to rebuild", ", ".join(map(str, existing)))
            return 1
        logger.warning("--force: overwriting %s", ", ".join(map(str, existing)))

    generation = load_config("generation", args.overrides)
    models = load_config("models", args.overrides)["models"]
    evaluation = load_config("dataset", args.overrides)["evaluation"]
    source_dir = resolve_path(generation["runs"]["evaluation"]["output_dir"])
    sources = {model_key: source_dir / model["output"] for model_key, model in models.items()}
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        logger.error("Caption files not found: %s", missing)
        return 1

    records, report, errors = consolidate(
        sources,
        resolve_path(evaluation["manifest"]),
        resolve_path(evaluation["references"]),
        generation["prompt"]["text"],
        evaluation["captions_per_recording"],
    )
    report = {"run_info": collect_run_info(Path(__file__).stem, seed=args.seed), **report}
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    if errors:
        for error in errors[:20]:
            logger.error("%s", error)
        logger.error("%d problems; the caption dataset was not written (see report)", len(errors))
    else:
        write_jsonl(records, output)
        write_checksums([output, *sources.values()], checksum_path)
        checksums = {
            line.split(maxsplit=1)[1]: line.split()[0]
            for line in checksum_path.read_text(encoding="utf-8").splitlines()
        }
        report["output"] = {"path": f"{output_dir.name}/{OUTPUT_NAME}", "sha256": sha256sum(output)}
        report["checksums"] = checksums
    write_json(report, report_dir / "caption_validation.json")
    (report_dir / "caption_validation.md").write_text(report_markdown(report, checksums), encoding="utf-8")
    logger.info("Wrote caption_validation.{json,md} to %s", report_dir)
    if errors:
        return 1

    logger.info(
        "Wrote %d captions (%d recordings x %d models) to %s", len(records), report["recordings"], len(sources), output
    )
    for model_key, item in report["per_model"].items():
        logger.info("%s: %s, malformed types %s", model_key, item["normalization"], item["malformed_types"])
    logger.info("Recordings valid for all models: %d", report["valid_for_all_models"]["count"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
