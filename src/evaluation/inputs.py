"""Load and check evaluation inputs (E10-E13): model captions, human references and audio."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.clotho import read_jsonl, sha256sum
from src.metrics.preprocess import check_texts


def load_references(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_captions(path: str | Path) -> dict[str, dict[str, str]]:
    """`model_key` -> `sample_id` -> the caption to score (field `caption` of all_model_captions.jsonl)."""
    captions: dict[str, dict[str, str]] = defaultdict(dict)
    for record in read_jsonl(path):
        if record["sample_id"] in captions[record["model_key"]]:
            raise ValueError(f"Duplicate caption for {record['model_key']} {record['sample_id']}")
        captions[record["model_key"]][record["sample_id"]] = record["caption"]
    return dict(captions)


def check_condition_inputs(
    sample_ids: list[str], candidates: dict[str, Any], references: dict[str, Any], num_references: int
) -> list[str]:
    """Problems that make a scoring call invalid. Nothing is dropped: any problem stops the run."""
    errors = []
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("duplicate sample_ids")
    if missing := [sid for sid in sample_ids if sid not in candidates]:
        errors.append(f"{len(missing)} recordings without a candidate, e.g. {missing[:3]}")
    if extra := sorted(set(candidates) - set(sample_ids)):
        errors.append(f"{len(extra)} candidates for unknown recordings, e.g. {extra[:3]}")
    if missing := [sid for sid in sample_ids if sid not in references]:
        errors.append(f"{len(missing)} recordings without references, e.g. {missing[:3]}")
    if errors:
        return errors
    errors += check_texts([candidates[sid] for sid in sample_ids], "candidate")
    for sid in sample_ids:
        refs = references[sid]
        if not isinstance(refs, list) or len(refs) != num_references:
            found = len(refs) if isinstance(refs, list) else repr(refs)
            errors.append(f"{sid}: {found} references, expected {num_references}")
        else:
            errors += check_texts(refs, f"{sid} reference")
    return errors


def audio_paths(manifest: list[dict[str, Any]], root: str | Path, verify: bool = True) -> dict[str, Path]:
    """`sample_id` -> WAV path under the dataset root; with `verify`, every file must match its manifest SHA-256."""
    paths, errors = {}, []
    for record in manifest:
        path = Path(root) / record["audio_path"]
        if not path.is_file():
            errors.append(f"{record['sample_id']}: {path} not found")
        elif verify and sha256sum(path) != record["audio_sha256"]:
            errors.append(f"{record['sample_id']}: {path} does not match the manifest SHA-256")
        paths[record["sample_id"]] = path
    if errors:
        raise ValueError(f"{len(errors)} audio problems: {errors[:5]}")
    return paths
