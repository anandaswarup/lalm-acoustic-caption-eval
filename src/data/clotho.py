"""Load and validate the local Clotho v2.1 copy (E02)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf

EXPECTED_SAMPLE_RATE = 44100
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2

_SUBTYPE_WIDTH = {"PCM_U8": 1, "PCM_S8": 1, "PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8}
_CHUNK_BYTES = 1 << 20


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV as UTF-8, falling back to Latin-1 as the original release requires for some files."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text)))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _file_hash(path: str | Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def md5sum(path: str | Path) -> str:
    return _file_hash(path, "md5")


def sha256sum(path: str | Path) -> str:
    return _file_hash(path, "sha256")


def captions_csv(root: Path, split: str) -> Path:
    return root / "captions" / f"clotho_captions_{split}.csv"


def metadata_csv(root: Path, split: str) -> Path:
    return root / "metadata" / f"clotho_metadata_{split}.csv"


def dataset_manifest(root: Path, split: str) -> Path:
    return root / "manifests" / f"{split}.jsonl"


def zenodo_record_path(root: Path, source: str) -> Path:
    record_id = source.rstrip("/").split("/")[-1]
    return root / "provenance" / f"zenodo-record-{record_id}.json"


def load_split(root: str | Path, split: str) -> list[dict[str, Any]]:
    """Join the captions CSV, metadata CSV and dataset manifest on `file_name`.

    Each record has `file_name`, `audio_path` (relative to `root`), `captions` (CSV order), `metadata` (CSV row)
    and `manifest` (the dataset manifest record, or None if missing). Records are sorted by `file_name`.
    """
    root = Path(root)
    metadata = {row["file_name"]: row for row in read_csv_rows(metadata_csv(root, split))}
    manifest = {row["file_name"]: row for row in read_jsonl(dataset_manifest(root, split))}
    records = []
    for row in read_csv_rows(captions_csv(root, split)):
        name = row["file_name"]
        caption_keys = sorted(
            (key for key in row if key.startswith("caption_")), key=lambda key: int(key.split("_")[1])
        )
        records.append(
            {
                "file_name": name,
                "audio_path": f"audio/{split}/{name}",
                "captions": [row[key] for key in caption_keys],
                "metadata": metadata.get(name),
                "manifest": manifest.get(name),
            }
        )
    return sorted(records, key=lambda record: record["file_name"])


def check_audio(
    path: str | Path,
    expected_frames: int | None = None,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    channels: int = EXPECTED_CHANNELS,
    sample_width_bytes: int = EXPECTED_SAMPLE_WIDTH_BYTES,
) -> tuple[dict[str, Any], list[str]]:
    """Decode a whole audio file; return its properties and any problems found."""
    path = Path(path)
    try:
        info = sf.info(path)
        data, decoded_rate = sf.read(path, dtype="int16", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as error:
        return {}, [f"cannot decode: {error}"]
    properties = {
        "sample_rate": decoded_rate,
        "channels": data.shape[1],
        "sample_width_bytes": _SUBTYPE_WIDTH.get(info.subtype),
        "num_frames": data.shape[0],
        "duration_seconds": data.shape[0] / decoded_rate,
        "sha256": sha256sum(path),
    }
    problems = []
    if properties["sample_rate"] != sample_rate:
        problems.append(f"sample rate {properties['sample_rate']} != {sample_rate}")
    if properties["channels"] != channels:
        problems.append(f"channels {properties['channels']} != {channels}")
    if properties["sample_width_bytes"] != sample_width_bytes:
        problems.append(f"sample width {properties['sample_width_bytes']} ({info.subtype}) != {sample_width_bytes}")
    if properties["num_frames"] == 0:
        problems.append("no audio frames")
    if expected_frames is not None and properties["num_frames"] != expected_frames:
        problems.append(f"decoded {properties['num_frames']} frames, dataset manifest says {expected_frames}")
    return properties, problems


@dataclass
class SplitValidation:
    split: str
    records: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)
    csv_checksums: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> dict[str, Any]:
        durations = [record["audio"]["duration_seconds"] for record in self.records if record.get("audio")]
        return {
            "split": self.split,
            "recordings": len(self.records),
            "captions": sum(len(record["captions"]) for record in self.records),
            "decoded": len(durations),
            "duration_seconds": {
                "min": min(durations, default=None),
                "max": max(durations, default=None),
                "total": sum(durations),
            },
            "csv_checksums": self.csv_checksums,
            "errors": self.errors,
            "ok": self.ok,
        }


def _zenodo_md5s(record_path: Path) -> dict[str, str]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return {item["key"]: item["checksum"].removeprefix("md5:") for item in record.get("files", [])}


def validate_split(
    root: str | Path,
    split: str,
    expected_recordings: int | None,
    captions_per_recording: int,
    source: str,
    file_names: set[str] | None = None,
) -> SplitValidation:
    """Run every E02 check on one split.

    `expected_recordings=None` skips the count check; with `file_names`, only those recordings are decoded.
    """
    root = Path(root)
    records = load_split(root, split)
    result = SplitValidation(split=split, records=records)
    errors = result.errors

    # Release files against the Zenodo record checksums.
    zenodo = _zenodo_md5s(zenodo_record_path(root, source))
    for path in (captions_csv(root, split), metadata_csv(root, split)):
        expected, actual = zenodo.get(path.name), md5sum(path)
        result.csv_checksums[path.name] = {"md5": actual, "zenodo_md5": expected, "match": actual == expected}
        if actual != expected:
            errors.append(f"{path.name}: MD5 {actual} does not match Zenodo record ({expected})")

    # Recording counts and file names.
    names = [record["file_name"] for record in records]
    if expected_recordings is not None and len(names) != expected_recordings:
        errors.append(f"{len(names)} recordings in captions CSV, expected {expected_recordings}")
    if len(set(names)) != len(names):
        errors.append(f"{len(names) - len(set(names))} duplicate file names in captions CSV")
    on_disk = {path.name for path in (root / "audio" / split).glob("*.wav")}
    if missing := sorted(set(names) - on_disk):
        errors.append(f"{len(missing)} audio files missing, e.g. {missing[:3]}")
    if extra := sorted(on_disk - set(names)):
        errors.append(f"{len(extra)} audio files without captions, e.g. {extra[:3]}")

    for record in records:
        name, captions, manifest = record["file_name"], record["captions"], record["manifest"]
        if len(captions) != captions_per_recording or any(not caption.strip() for caption in captions):
            non_empty = sum(bool(caption.strip()) for caption in captions)
            errors.append(f"{name}: {non_empty} non-empty captions, expected {captions_per_recording}")
        if record["metadata"] is None:
            errors.append(f"{name}: no metadata row")
        if manifest is None:
            errors.append(f"{name}: not in dataset manifest")
        elif manifest["captions"] != captions:
            errors.append(f"{name}: dataset manifest captions differ from captions CSV")
        if (file_names is None or name in file_names) and name in on_disk:
            expected_frames = manifest["num_frames"] if manifest else None
            record["audio"], problems = check_audio(root / record["audio_path"], expected_frames)
            errors.extend(f"{name}: {problem}" for problem in problems)

    return result


def provenance(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Dataset version, DOI, source, licence and the Zenodo checksums of the caption and metadata files."""
    root = Path(root)
    record = json.loads(zenodo_record_path(root, config["source"]).read_text(encoding="utf-8"))
    files = {
        item["key"]: {"size": item["size"], "md5": item["checksum"].removeprefix("md5:")}
        for item in record.get("files", [])
    }
    return {
        "name": config["name"],
        "version": config["version"],
        "doi": config["doi"],
        "source": config["source"],
        "zenodo_record_version": record.get("metadata", {}).get("version"),
        "zenodo_record_modified": record.get("modified"),
        "license": config["license"],
        "license_file_sha256": sha256sum(root / "LICENSE"),
        "zenodo_files": files,
    }
