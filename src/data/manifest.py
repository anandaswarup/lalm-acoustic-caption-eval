"""Build, check and checksum the frozen evaluation manifests (E03), and draw seeded samples of recordings."""

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


def select_sample(file_names: Iterable[str], num_recordings: int, seed: int) -> list[str]:
    """Draw `num_recordings` names with a seeded RNG; independent of input order."""
    population = sorted(file_names)
    return sorted(random.Random(seed).sample(population, num_recordings))


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
