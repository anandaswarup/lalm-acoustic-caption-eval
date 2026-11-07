"""Clotho download, verification and indexing (no network; synthetic files)."""

import hashlib
import json
import wave

import pytest

from src.data.clotho_download import (
    build_manifest,
    download_file,
    manifest_text,
    md5sum,
    read_rows,
    split_summary,
    target_path,
    wanted_files,
)

RECORD = {
    "files": [
        {"key": "clotho_audio_development.7z"},
        {"key": "clotho_audio_evaluation.7z"},
        {"key": "clotho_captions_evaluation.csv"},
        {"key": "clotho_metadata_development.csv"},
        {"key": "LICENSE"},
    ]
}


def test_wanted_files_and_targets(tmp_path):
    keys = [item["key"] for item in wanted_files(RECORD, ["evaluation"])]
    assert keys == ["clotho_audio_evaluation.7z", "clotho_captions_evaluation.csv", "clotho_metadata_development.csv",
                    "LICENSE"]  # fmt: skip
    assert target_path(tmp_path, "clotho_audio_evaluation.7z") == tmp_path / "archives" / "clotho_audio_evaluation.7z"
    assert target_path(tmp_path, "clotho_captions_evaluation.csv").parent.name == "captions"
    assert target_path(tmp_path, "clotho_metadata_evaluation.csv").parent.name == "metadata"
    assert target_path(tmp_path, "LICENSE") == tmp_path / "LICENSE"


def test_download_file_verifies_an_existing_file(tmp_path):
    (tmp_path / "LICENSE").write_bytes(b"licence text")
    item = {"key": "LICENSE", "size": 12, "checksum": "md5:" + hashlib.md5(b"licence text").hexdigest()}  # noqa: S324
    assert download_file(item, tmp_path, lambda message: None) == tmp_path / "LICENSE"
    assert md5sum(tmp_path / "LICENSE") == item["checksum"][4:]
    with pytest.raises(RuntimeError, match="MD5"):
        download_file({**item, "checksum": "md5:0"}, tmp_path, lambda message: None)
    with pytest.raises(RuntimeError, match="size"):
        download_file({**item, "size": 13}, tmp_path, lambda message: None)


def write_wav(path, frames: int, truncate: bool = False):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\0\1" * frames)
    if truncate:
        path.write_bytes(path.read_bytes()[:-10])


def dataset(tmp_path, split="evaluation"):
    for folder in ("captions", "metadata", f"audio/{split}"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    names = ["b.wav", "a é.wav"]
    header = "file_name," + ",".join(f"caption_{i}" for i in range(1, 6)) + "\n"
    rows = "".join(f'"{n}",' + ",".join(f"cap {i} of {n}" for i in range(1, 6)) + "\n" for n in names)
    (tmp_path / "captions" / f"clotho_captions_{split}.csv").write_text(header + rows, encoding="utf-8")
    metadata = "file_name,keywords\n" + "".join(f'"{n}",café\n' for n in names)
    (tmp_path / "metadata" / f"clotho_metadata_{split}.csv").write_bytes(metadata.encode("latin-1"))
    for i, name in enumerate(names):
        write_wav(tmp_path / "audio" / split / name, 100 * (i + 1))
    return names


def test_build_manifest(tmp_path):
    names = dataset(tmp_path)
    assert read_rows(tmp_path / "metadata" / "clotho_metadata_evaluation.csv")[0]["keywords"] == "café"  # Latin-1
    records = build_manifest(tmp_path, "evaluation", expected=2)
    assert [r["file_name"] for r in records] == names  # caption-CSV order
    first = records[0]
    assert first["audio_path"] == "audio/evaluation/b.wav" and first["captions"][4] == "cap 5 of b.wav"
    assert (first["sample_rate"], first["channels"], first["num_frames"]) == (44100, 1, 100)
    text = manifest_text(records)
    assert "a é.wav" in text and json.loads(text.splitlines()[1])["metadata"]["keywords"] == "café"
    assert split_summary(records)["audio_files"] == 2 and split_summary(records)["captions"] == 10
    with pytest.raises(ValueError, match="expected 3"):
        build_manifest(tmp_path, "evaluation", expected=3)


def test_build_manifest_rejects_misaligned_or_truncated_audio(tmp_path):
    dataset(tmp_path)
    (tmp_path / "audio" / "evaluation" / "b.wav").unlink()
    with pytest.raises(ValueError, match="align"):
        build_manifest(tmp_path, "evaluation")
    write_wav(tmp_path / "audio" / "evaluation" / "b.wav", 100, truncate=True)
    with pytest.raises(ValueError, match="truncated"):
        build_manifest(tmp_path, "evaluation")
