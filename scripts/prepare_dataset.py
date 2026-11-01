"""Validate Clotho v2.1 and build the frozen evaluation and pilot manifests (E02-E03).

Writes the manifests, a copy of the Clotho caption licence and SHA256SUMS to the manifest directory, and the
validation and manifest reports to the report directory. Existing manifests are never overwritten without --force.
"""

import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import load_split, provenance, validate_split  # noqa: E402
from src.data.manifest import (  # noqa: E402
    build_eval_manifest,
    build_pilot_manifest,
    check_manifest,
    exclude_shared_sources,
    known_sound_id,
    select_pilot,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

LICENSE_COPY = "CLOTHO_LICENSE"
CHECKSUM_FILE = "SHA256SUMS"


def log_errors(logger, label, errors, limit=20):
    for error in errors[:limit]:
        logger.error("%s: %s", label, error)
    if len(errors) > limit:
        logger.error("%s: ... and %d more (see report)", label, len(errors) - limit)


def write_markdown(path, title, sections):
    lines = [f"# {title}", ""]
    for heading, rows in sections:
        lines += [f"## {heading}", "", "| Item | Value |", "|---|---|"]
        lines += [f"| {key} | {value} |" for key, value in rows]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing frozen manifests")
    parser.add_argument("--check", action="store_true", help="only verify existing manifests against SHA256SUMS")
    parser.add_argument("--report-dir", default=str(OUTPUT_DIR / "reports"), help="default: %(default)s")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    config = load_config("dataset", args.overrides)
    evaluation, pilot = config["evaluation"], config["pilot"]

    eval_path, refs_path, pilot_path = (
        resolve_path(path) for path in (evaluation["manifest"], evaluation["references"], pilot["manifest"])
    )
    manifest_dir = eval_path.parent
    if {refs_path.parent, pilot_path.parent} != {manifest_dir}:
        logger.error("All manifests must share one directory so SHA256SUMS can list them")
        return 1
    checksum_path = manifest_dir / CHECKSUM_FILE

    if args.check:
        errors = verify_checksums(checksum_path)
        log_errors(logger, "check", errors)
        if not errors:
            logger.info("All files in %s match", checksum_path)
        return 1 if errors else 0

    outputs = [eval_path, refs_path, pilot_path, checksum_path]
    if existing := [path for path in outputs if path.exists()]:
        if not args.force:
            logger.error("Frozen manifests already exist (%s); use --force to rebuild", ", ".join(map(str, existing)))
            return 1
        logger.warning("--force: overwriting %s", ", ".join(map(str, existing)))

    root = resolve_path(config["root"])
    if not root.is_dir():
        logger.error("Dataset root not found: %s (set CLOTHO_ROOT in .env)", root)
        return 1
    logger.info("Dataset root: %s", root)

    # E02: validate the evaluation split (decoding every file) and the pilot recordings.
    logger.info("Validating the %s split", evaluation["split"])
    eval_check = validate_split(
        root,
        evaluation["split"],
        evaluation["expected_recordings"],
        evaluation["captions_per_recording"],
        config["source"],
    )
    # Draw the pilot only from recordings that share no Freesound source or file name with the evaluation split.
    pilot_pool, excluded = exclude_shared_sources(load_split(root, pilot["split"]), eval_check.records)
    unknown_source = {
        "evaluation": sum(known_sound_id(r["metadata"]["sound_id"]) is None for r in eval_check.records),
        "pilot_pool": sum(known_sound_id(r["metadata"]["sound_id"]) is None for r in pilot_pool),
    }
    excluded_summary = [
        {"file_name": record["file_name"], "sound_id": (record["metadata"] or {}).get("sound_id")}
        for record in excluded
    ]
    logger.info(
        "Pilot pool: %d %s recordings (%d excluded as shared with %s)",
        len(pilot_pool),
        pilot["split"],
        len(excluded),
        evaluation["split"],
    )
    pilot_names = select_pilot((record["file_name"] for record in pilot_pool), pilot["num_recordings"], pilot["seed"])
    logger.info("Validating %d pilot recordings from the %s split", len(pilot_names), pilot["split"])
    pilot_check = validate_split(
        root, pilot["split"], None, evaluation["captions_per_recording"], config["source"], set(pilot_names)
    )

    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    validation_report = {
        "run": collect_run_info(Path(__file__).stem, seed=args.seed),
        "provenance": provenance(root, config),
        "evaluation": eval_check.summary(),
        "pilot": {
            **pilot_check.summary(),
            "pool": len(pilot_pool),
            "excluded_shared_with_evaluation": excluded_summary,
            "unknown_freesound_source": unknown_source,
            "selected": pilot_names,
        },
    }
    write_json(validation_report, report_dir / "dataset_validation.json")
    eval_summary = validation_report["evaluation"]
    write_markdown(
        report_dir / "dataset_validation.md",
        "Clotho v2.1 validation report",
        [
            ("Provenance", [(key, validation_report["provenance"][key]) for key in ("version", "doi", "source")]),
            (
                f"Split: {evaluation['split']}",
                [
                    ("Recordings", eval_summary["recordings"]),
                    ("Captions", eval_summary["captions"]),
                    ("Decoded", eval_summary["decoded"]),
                    ("Duration min / max (s)", "{min:.2f} / {max:.2f}".format(**eval_summary["duration_seconds"])),
                    ("Total duration (h)", f"{eval_summary['duration_seconds']['total'] / 3600:.3f}"),
                    ("CSV MD5s match Zenodo", all(item["match"] for item in eval_summary["csv_checksums"].values())),
                    ("Errors", len(eval_summary["errors"])),
                ],
            ),
            (
                f"Pilot ({pilot['split']})",
                [
                    ("Pool after excluding shared sources", len(pilot_pool)),
                    ("Excluded (shared with evaluation)", ", ".join(f"`{r['file_name']}`" for r in excluded_summary)),
                    (
                        "Unknown Freesound source (evaluation / pilot pool)",
                        "{evaluation} / {pilot_pool}".format(**unknown_source),
                    ),
                    ("Recordings", len(pilot_names)),
                    ("Errors", len(pilot_check.errors)),
                ],
            ),
        ],
    )
    logger.info("Wrote %s", report_dir / "dataset_validation.json")

    log_errors(logger, evaluation["split"], eval_check.errors)
    log_errors(logger, "pilot", pilot_check.errors)
    if not (eval_check.ok and pilot_check.ok):
        logger.error("Validation failed; no manifests written")
        return 1
    logger.info("Validation passed: %d recordings, %d captions", eval_summary["recordings"], eval_summary["captions"])

    # E03: build, check and write the frozen manifests.
    manifest, references = build_eval_manifest(eval_check.records, evaluation["split"])
    pilot_manifest = build_pilot_manifest(
        [record for record in pilot_check.records if record["file_name"] in set(pilot_names)], pilot["split"]
    )
    errors = check_manifest(
        manifest, references, evaluation["expected_recordings"], evaluation["captions_per_recording"]
    )
    errors += [f"pilot: {error}" for error in check_manifest(pilot_manifest, None, pilot["num_recordings"], 0)]
    eval_names = {r["file_name"] for r in manifest}
    eval_ids = {known_sound_id(r["sound_id"]) for r in manifest} - {None}
    shared = [r for r in pilot_manifest if r["file_name"] in eval_names or known_sound_id(r["sound_id"]) in eval_ids]
    if overlap := [r["file_name"] for r in shared]:
        errors.append(f"pilot overlaps the evaluation set: {overlap[:3]}")
    log_errors(logger, "manifest", errors)
    if errors:
        return 1

    write_jsonl(manifest, eval_path)
    write_json(references, refs_path)
    write_jsonl(pilot_manifest, pilot_path)
    shutil.copyfile(root / "LICENSE", manifest_dir / LICENSE_COPY)
    write_checksums([eval_path, refs_path, pilot_path, manifest_dir / LICENSE_COPY], checksum_path)
    checksums = dict(
        reversed(line.split(maxsplit=1)) for line in checksum_path.read_text(encoding="utf-8").splitlines()
    )

    manifest_report = {
        "run": validation_report["run"],
        "evaluation": {
            "recordings": len(manifest),
            "captions": sum(len(captions) for captions in references.values()),
            "first_sample_id": manifest[0]["sample_id"],
            "last_sample_id": manifest[-1]["sample_id"],
            "order_by": evaluation["order_by"],
            "audio_licenses": dict(Counter(record["audio_license"] for record in manifest).most_common()),
        },
        "pilot": {
            "recordings": len(pilot_manifest),
            "split": pilot["split"],
            "seed": pilot["seed"],
            "pool": len(pilot_pool),
            "excluded_shared_with_evaluation": len(excluded),
            "unknown_freesound_source_in_pool": unknown_source["pilot_pool"],
        },
        "sha256": checksums,
    }
    write_json(manifest_report, report_dir / "manifest_report.json")
    write_markdown(
        report_dir / "manifest_report.md",
        "Frozen manifest report",
        [
            (
                "Evaluation",
                [
                    (key, manifest_report["evaluation"][key])
                    for key in ("recordings", "captions", "first_sample_id", "last_sample_id", "order_by")
                ],
            ),
            ("Pilot", list(manifest_report["pilot"].items())),
            ("SHA-256", [(f"`{name}`", f"`{digest}`") for name, digest in checksums.items()]),
            ("Audio licences (evaluation)", list(manifest_report["evaluation"]["audio_licenses"].items())),
        ],
    )
    logger.info("Wrote manifests and %s to %s", CHECKSUM_FILE, manifest_dir)
    logger.info("Wrote %s", report_dir / "manifest_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
