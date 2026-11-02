"""Build, check and checksum the frozen evaluation and pilot manifests (E03)."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.data.clotho import sha256sum


def _manifest_record(sample_id: str, split: str, record: dict[str, Any]) -> dict[str, Any]:
    audio, metadata = record["audio"], record["metadata"]
    return {
        "sample_id": sample_id,
        "file_name": record["file_name"],
        "audio_path": record["audio_path"],
        "split": split,
        "sample_rate": audio["sample_rate"],
        "channels": audio["channels"],
        "num_frames": audio["num_frames"],
        "duration_seconds": audio["duration_seconds"],
        "audio_sha256": audio["sha256"],
        "sound_id": metadata["sound_id"],
        "sound_link": metadata["sound_link"],
        "audio_license": metadata["license"],
    }


def build_eval_manifest(
    records: list[dict[str, Any]], split: str = "evaluation", prefix: str = "clotho_eval"
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Number validated records in `file_name` order; return the manifest and `sample_id` -> captions."""
    records = sorted(records, key=lambda record: record["file_name"])
    width = max(4, len(str(len(records) - 1)))
    manifest, references = [], {}
    for index, record in enumerate(records):
        sample_id = f"{prefix}_{index:0{width}d}"
        manifest.append(_manifest_record(sample_id, split, record))
        references[sample_id] = list(record["captions"])
    return manifest, references


# Clotho metadata uses "Not found" when the Freesound source is unknown.
UNKNOWN_SOUND_IDS = {"", "Not found"}


def known_sound_id(sound_id: str | None) -> str | None:
    """Return the Freesound ID, or None for Clotho's unknown-source placeholders."""
    return None if sound_id is None or sound_id.strip() in UNKNOWN_SOUND_IDS else sound_id


def _sound_id(record: dict[str, Any]) -> str | None:
    return known_sound_id((record.get("metadata") or {}).get("sound_id"))


def exclude_shared_sources(
    candidates: list[dict[str, Any]], held_out: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split `candidates` into those sharing no Freesound source or file name with `held_out`, and the rest.

    Clotho cuts some Freesound sounds into excerpts that land in different splits, so file names alone miss overlap.
    Records whose source is unknown can only be matched by file name.
    """
    held_ids = {_sound_id(record) for record in held_out} - {None}
    held_names = {record["file_name"] for record in held_out}
    kept, excluded = [], []
    for record in candidates:
        sound_id = _sound_id(record)
        shared = (sound_id is not None and sound_id in held_ids) or record["file_name"] in held_names
        (excluded if shared else kept).append(record)
    return kept, excluded


def select_pilot(file_names: Iterable[str], num_recordings: int, seed: int) -> list[str]:
    """Draw `num_recordings` file names with a seeded RNG; independent of input order."""
    population = sorted(file_names)
    return sorted(random.Random(seed).sample(population, num_recordings))


def build_pilot_manifest(
    records: list[dict[str, Any]], split: str = "validation", prefix: str = "clotho_pilot"
) -> list[dict[str, Any]]:
    """Pilot records carry their captions inline; the pilot has no separate references file."""
    records = sorted(records, key=lambda record: record["file_name"])
    width = max(2, len(str(len(records) - 1)))
    return [
        {**_manifest_record(f"{prefix}_{index:0{width}d}", split, record), "captions": list(record["captions"])}
        for index, record in enumerate(records)
    ]


def check_manifest(
    manifest: list[dict[str, Any]],
    references: dict[str, list[str]] | None,
    expected_recordings: int,
    captions_per_recording: int,
) -> list[str]:
    """Return problems with uniqueness, completeness and manifest/reference agreement."""
    errors = []
    sample_ids = [record["sample_id"] for record in manifest]
    file_names = [record["file_name"] for record in manifest]
    if len(manifest) != expected_recordings:
        errors.append(f"{len(manifest)} recordings, expected {expected_recordings}")
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("duplicate sample_id values")
    if len(set(file_names)) != len(file_names):
        errors.append("duplicate file_name values")
    if file_names != sorted(file_names):
        errors.append("records are not sorted by file_name")
    if references is not None:
        if set(references) != set(sample_ids):
            errors.append("reference sample_ids do not match manifest sample_ids")
        bad = [key for key, captions in references.items() if len(captions) != captions_per_recording]
        if bad:
            errors.append(f"{len(bad)} samples without {captions_per_recording} references, e.g. {bad[:3]}")
        total = sum(len(captions) for captions in references.values())
        if total != expected_recordings * captions_per_recording:
            errors.append(f"{total} reference captions, expected {expected_recordings * captions_per_recording}")
    return errors


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.writelines(_dump(record) + "\n" for record in records)
    return path


def write_json(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return path


def write_checksums(paths: Iterable[str | Path], out: str | Path) -> Path:
    """Write `<sha256>  <name>` lines (sha256sum format) for files in or below the directory of `out`.

    Names are relative to that directory (`a.jsonl`, `sub/b.jsonl`), so `verify_checksums` finds them again.
    """
    out = Path(out)
    names = {Path(path).resolve().relative_to(out.parent.resolve()).as_posix(): path for path in paths}
    lines = [f"{sha256sum(names[name])}  {name}\n" for name in sorted(names)]
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        stream.writelines(lines)
    return out


def verify_checksums(checksum_file: str | Path) -> list[str]:
    """Return problems for files listed in `checksum_file` that are missing or changed."""
    checksum_file = Path(checksum_file)
    if not checksum_file.exists():
        return [f"{checksum_file.name} not found"]
    errors = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        path = checksum_file.parent / name.strip()
        if not path.exists():
            errors.append(f"{name}: missing")
        elif sha256sum(path) != expected:
            errors.append(f"{name}: checksum changed")
    return errors
