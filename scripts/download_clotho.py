"""Download, verify, extract and index Clotho v2.1 into CLOTHO_ROOT (before E02).

Fetches the Zenodo record snapshot (once), downloads the release files with resume and MD5 verification (every
caption and metadata CSV, the licence, and the audio archives of --splits), extracts the archives with bsdtar, and
writes manifests/<split>.jsonl and provenance/validation.json (see src/data/clotho_download.py). `make prepare-data`
then builds the frozen manifests from this folder. --check rebuilds the manifests of the splits present and compares
them with the existing files, without downloading or writing anything.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho_download import (  # noqa: E402
    EXPECTED,
    build_manifest,
    download_file,
    extract,
    fetch_record,
    manifest_text,
    split_summary,
    validation_summary,
    wanted_files,
)
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--root", help="dataset folder (default: CLOTHO_ROOT via configs/dataset.yaml)")
    parser.add_argument("--splits", nargs="+", default=["evaluation"], choices=list(EXPECTED),
                        help="audio splits to download and index (default: evaluation)")  # fmt: skip
    parser.add_argument("--check", action="store_true", help="only compare rebuilt manifests with the existing ones")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    config = load_config("dataset", args.overrides)
    root = resolve_path(args.root or config["root"])
    splits = [split for split in EXPECTED if split in args.splits]  # release order

    if args.check:
        mismatched = []
        for split in splits:
            existing = root / "manifests" / f"{split}.jsonl"
            rebuilt = manifest_text(build_manifest(root, split, EXPECTED[split])).encode("utf-8")
            same = existing.is_file() and existing.read_bytes() == rebuilt
            logger.info("%s: rebuilt manifest %s the existing file", split, "matches" if same else "DIFFERS from")
            mismatched += [] if same else [split]
        return 1 if mismatched else 0

    if shutil.which("curl") is None or shutil.which("bsdtar") is None:
        logger.error("curl and bsdtar are required")
        return 1
    root.mkdir(parents=True, exist_ok=True)
    record = fetch_record(root, config["source"], config["version"])
    logger.info("Zenodo record %s, version %s, modified %s", config["source"], record["metadata"]["version"],
                record.get("modified"))  # fmt: skip
    for item in sorted(wanted_files(record, splits), key=lambda item: item["size"]):
        download_file(item, root, logger.info)

    summaries = {}
    for split in splits:
        audio_dir = root / "audio" / split
        if len(list(audio_dir.glob("*.wav"))) != EXPECTED[split]:
            logger.info("Extracting %s", split)
            extract(root, split)
        records = build_manifest(root, split, EXPECTED[split])
        (root / "manifests").mkdir(exist_ok=True)
        (root / "manifests" / f"{split}.jsonl").write_text(manifest_text(records), encoding="utf-8")
        summaries[split] = split_summary(records)
        logger.info("Validated %s: %s", split, summaries[split])
    summary = validation_summary(config["version"], config["source"], summaries)
    (root / "provenance" / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Clotho %s ready in %s (%s)", config["version"], root, ", ".join(splits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
