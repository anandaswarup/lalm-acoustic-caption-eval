"""Download, verify, extract and index the Clotho v2.1 release (before E02).

Builds the local dataset folder that `CLOTHO_ROOT` points to:

    provenance/zenodo-record-<id>.json   Zenodo record snapshot (file list, sizes, MD5); fetched once, then reused
    LICENSE, captions/*.csv, metadata/*.csv, archives/clotho_audio_<split>.7z   release files, MD5-verified
    audio/<split>/*.wav                  extracted with bsdtar
    manifests/<split>.jsonl              one record per audio file: captions, audio properties, metadata row
    provenance/validation.json           per-split counts, duration and audio formats

Downloads resume after interruption; archives are fetched in 256 MB ranges in parallel (Zenodo is slow per
connection). The manifest format is the one the dataset was originally prepared with, so rebuilding a manifest from
the same files gives the same bytes.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import shutil
import subprocess
import time
import urllib.request
import wave
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED = {"development": 3839, "validation": 1045, "evaluation": 1045}
RANGE_BYTES = 256 * 1024 * 1024
CURL = ["curl", "-fL", "--connect-timeout", "60", "--speed-limit", "1024", "--speed-time", "120", "--silent",
        "--show-error"]  # fmt: skip


def record_path(root: Path, source: str) -> Path:
    return root / "provenance" / f"zenodo-record-{source.rstrip('/').split('/')[-1]}.json"


def fetch_record(root: Path, source: str, version: str) -> dict[str, Any]:
    """The Zenodo record snapshot: read from provenance/ if present, else fetched from the API and saved."""
    path = record_path(root, source)
    if not path.is_file():
        record_id = source.rstrip("/").split("/")[-1]
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{record_id}", timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["metadata"]["version"] != version:
        raise ValueError(f"Zenodo record version {record['metadata']['version']} != {version}")
    return record


def target_path(root: Path, key: str) -> Path:
    folder = (
        "archives"
        if key.endswith(".7z")
        else "captions"
        if "captions" in key
        else "metadata"
        if "metadata" in key
        else ""
    )
    return root / folder / key


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - Zenodo publishes MD5 checksums
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wanted_files(record: dict[str, Any], splits: list[str]) -> list[dict[str, Any]]:
    """Release files to download: every CSV and the licence, and the audio archives of `splits`."""
    return [
        item
        for item in record["files"]
        if not item["key"].endswith(".7z") or item["key"].removesuffix(".7z").split("_")[-1] in splits
    ]


def _download_range(item: dict[str, Any], start: int, end: int, part: Path) -> None:
    size = end - start + 1
    for attempt in range(9):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            raise RuntimeError(f"{part}: larger than its range")
        if have == size:
            return
        with part.open("ab") as stream:  # append only the missing bytes
            result = subprocess.run([*CURL, "--range", f"{start + have}-{end}", item["links"]["self"]], stdout=stream)
        if result.returncode and attempt == 8:
            result.check_returncode()
        if result.returncode:
            time.sleep(15)
    if part.stat().st_size != size:
        raise RuntimeError(f"{part}: range size mismatch")


def _download_archive(item: dict[str, Any], target: Path, workers: int) -> Path:
    """Parallel ranged download into `<target>.part` (resumable); returns the partial file."""
    partial = target.with_name(target.name + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    ranges = []
    for start in range(have, item["size"], RANGE_BYTES):
        end = min(start + RANGE_BYTES, item["size"]) - 1
        ranges.append((start, end, target.with_name(f"{target.name}.range-{start}-{end}")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda r: _download_range(item, *r), ranges))
    with partial.open("ab") as stream:
        for _, _, part in ranges:
            with part.open("rb") as source:
                shutil.copyfileobj(source, stream, 8 * 1024 * 1024)
            part.unlink()
    return partial


def download_file(item: dict[str, Any], root: Path, log: Callable[[str], None], workers: int = 8) -> Path:
    """Download one release file if needed (resumable) and verify its size and MD5."""
    target = target_path(root, item["key"])
    target.parent.mkdir(parents=True, exist_ok=True)
    path = target
    if not target.exists():
        log(f"Downloading {item['key']} ({item['size'] / 1e6:.0f} MB)")
        if item["key"].endswith(".7z"):
            path = _download_archive(item, target, workers)
        else:
            path = target.with_name(target.name + ".part")
            subprocess.run([*CURL, "--retry", "8", "--retry-delay", "15", "-C", "-", item["links"]["self"], "-o",
                            str(path)], check=True)  # fmt: skip
    if path.stat().st_size != item["size"]:
        raise RuntimeError(f"{item['key']}: size {path.stat().st_size} != {item['size']}")
    if "md5:" + md5sum(path) != item["checksum"]:
        raise RuntimeError(f"{item['key']}: MD5 does not match the Zenodo record")
    if path != target:
        path.rename(target)
    log(f"Verified {item['key']}")
    return target


def extract(root: Path, split: str) -> None:
    (root / "audio").mkdir(exist_ok=True)
    subprocess.run(["bsdtar", "-xf", str(root / "archives" / f"clotho_audio_{split}.7z"), "-C", str(root / "audio")],
                   check=True)  # fmt: skip


def read_rows(path: Path) -> list[dict[str, str]]:
    """CSV rows; the release metadata contains Latin-1 characters, kept as in the source bytes."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text)))


def build_manifest(root: Path, split: str, expected: int | None = None) -> list[dict[str, Any]]:
    """One record per audio file, in caption-CSV order, after checking that captions, metadata and audio align and
    that every WAV's PCM payload is complete."""
    captions = read_rows(root / "captions" / f"clotho_captions_{split}.csv")
    metadata = read_rows(root / "metadata" / f"clotho_metadata_{split}.csv")
    names = [row["file_name"] for row in captions]
    meta = {row["file_name"]: row for row in metadata}
    audio = {path.name: path for path in (root / "audio" / split).glob("*.wav")}
    if len(names) != len(set(names)) or (expected is not None and len(names) != expected):
        raise ValueError(f"{split}: {len(names)} caption rows, {len(set(names))} distinct, expected {expected}")
    if len(meta) != len(metadata) or set(names) != set(meta) or set(names) != set(audio):
        raise ValueError(f"{split}: captions, metadata and audio files do not align")
    records = []
    for row in captions:
        name = row["file_name"]
        with wave.open(str(audio[name]), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            channels, width = wav.getnchannels(), wav.getsampwidth()
            if not (frames > 0 and rate > 0) or len(wav.readframes(frames)) != frames * channels * width:
                raise ValueError(f"{split}/{name}: empty or truncated audio")
        texts = [row[f"caption_{i}"] for i in range(1, 6)]
        if not all(text.strip() for text in texts):
            raise ValueError(f"{split}/{name}: empty caption")
        records.append(
            {
                "file_name": name,
                "audio_path": audio[name].relative_to(root).as_posix(),
                "split": split,
                "captions": texts,
                "sample_rate": rate,
                "channels": channels,
                "sample_width_bytes": width,
                "num_frames": frames,
                "duration_seconds": frames / rate,
                "metadata": meta[name],
            }
        )
    return records


def manifest_text(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio_files": len(records),
        "captions": len(records) * 5,
        "duration_hours": sum(r["duration_seconds"] for r in records) / 3600,
        "sample_rates": sorted({r["sample_rate"] for r in records}),
        "channels": sorted({r["channels"] for r in records}),
        "sample_width_bytes": sorted({r["sample_width_bytes"] for r in records}),
    }


def validation_summary(version: str, source: str, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": version,
        "source": source,
        "splits": summaries,
        "verified_at_utc": datetime.now(UTC).isoformat(),
    }
