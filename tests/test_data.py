"""E02-E03 on a small synthetic Clotho-like fixture (CI has no copy of Clotho)."""

import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from src.data.clotho import check_audio, load_split, validate_split
from src.data.manifest import (
    build_eval_manifest,
    build_pilot_manifest,
    check_manifest,
    exclude_shared_sources,
    select_pilot,
    verify_checksums,
    write_checksums,
)
from src.utils.paths import REPO_ROOT

SOURCE = "https://zenodo.org/records/4783391"
SPLITS = {"evaluation": ["c.wav", "a sound.wav", "b.wav"], "validation": ["v1.wav", "v2.wav", "v3.wav", "v4.wav"]}


def write_wav(path, seconds=0.1, sample_rate=44100, channels=1, subtype="PCM_16"):
    frames = int(seconds * sample_rate)
    data = np.zeros((frames, channels), dtype=np.int16)
    sf.write(path, data, sample_rate, subtype=subtype)
    return frames


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_zenodo_record(root):
    files = [
        {"key": path.name, "size": path.stat().st_size, "checksum": f"md5:{md5(path)}"}
        for path in sorted((root / "captions").glob("*.csv")) + sorted((root / "metadata").glob("*.csv"))
    ]
    record = {"modified": "2021-05-21", "metadata": {"version": "2.1"}, "files": files}
    (root / "provenance" / "zenodo-record-4783391.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def clotho(tmp_path):
    root = tmp_path / "clotho"
    for name in ("captions", "metadata", "manifests", "provenance"):
        (root / name).mkdir(parents=True)
    for split, names in SPLITS.items():
        (root / "audio" / split).mkdir(parents=True)
        captions = {name: [f"{name} caption {i}." for i in range(1, 6)] for name in names}
        frames = {
            name: write_wav(root / "audio" / split / name, seconds=0.1 + 0.05 * i) for i, name in enumerate(names)
        }
        write_csv(
            root / "captions" / f"clotho_captions_{split}.csv",
            ["file_name"] + [f"caption_{i}" for i in range(1, 6)],
            [[name, *captions[name]] for name in names],
        )
        write_csv(
            root / "metadata" / f"clotho_metadata_{split}.csv",
            ["file_name", "keywords", "sound_id", "sound_link", "start_end_samples", "manufacturer", "license"],
            [[name, "kw", f"{split}-{i}", f"https://fs.org/{i}", "", "x", "cc0"] for i, name in enumerate(names)],
        )
        with (root / "manifests" / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for name in names:
                stream.write(json.dumps({"file_name": name, "captions": captions[name], "num_frames": frames[name]}))
                stream.write("\n")
    (root / "LICENSE").write_text("Copyright notice\n", encoding="utf-8")
    write_zenodo_record(root)
    return root


def validate(root, split="evaluation", expected=3, file_names=None):
    return validate_split(root, split, expected, 5, SOURCE, file_names)


def test_load_split_joins_and_sorts(clotho):
    records = load_split(clotho, "evaluation")
    assert [record["file_name"] for record in records] == ["a sound.wav", "b.wav", "c.wav"]
    assert records[0]["audio_path"] == "audio/evaluation/a sound.wav"
    assert records[0]["captions"][0] == "a sound.wav caption 1."
    assert records[0]["metadata"]["license"] == "cc0"


def test_validate_split_passes(clotho):
    result = validate(clotho)
    assert result.ok, result.errors
    summary = result.summary()
    assert (summary["recordings"], summary["captions"], summary["decoded"]) == (3, 15, 3)
    assert all(item["match"] for item in summary["csv_checksums"].values())


def test_validate_split_wrong_count(clotho):
    assert any("expected 4" in error for error in validate(clotho, expected=4).errors)


def test_validate_split_empty_caption(clotho):
    path = clotho / "captions" / "clotho_captions_evaluation.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("b.wav caption 3.", ""), encoding="utf-8")
    write_zenodo_record(clotho)
    errors = validate(clotho).errors
    assert any("b.wav: 4 non-empty captions" in error for error in errors)
    assert any("b.wav: dataset manifest captions differ" in error for error in errors)


def test_validate_split_checksum_mismatch(clotho):
    path = clotho / "captions" / "clotho_captions_evaluation.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("c.wav caption 1.", "changed"), encoding="utf-8")
    assert any("does not match Zenodo" in error for error in validate(clotho).errors)


def test_validate_split_bad_audio(clotho):
    write_wav(clotho / "audio" / "evaluation" / "b.wav", sample_rate=16000, channels=2)
    write_wav(clotho / "audio" / "evaluation" / "extra.wav")
    errors = validate(clotho).errors
    assert any("b.wav: sample rate 16000" in error for error in errors)
    assert any("b.wav: channels 2" in error for error in errors)
    assert any("b.wav: decoded" in error for error in errors)
    assert any("without captions" in error for error in errors)


def test_check_audio_undecodable(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio")
    properties, problems = check_audio(path)
    assert properties == {} and problems[0].startswith("cannot decode")


def test_validate_split_decodes_only_selected(clotho):
    result = validate(clotho, split="validation", expected=None, file_names={"v2.wav"})
    assert result.ok and result.summary()["decoded"] == 1


def test_build_eval_manifest(clotho):
    manifest, references = build_eval_manifest(validate(clotho).records)
    assert [record["sample_id"] for record in manifest] == ["clotho_eval_0000", "clotho_eval_0001", "clotho_eval_0002"]
    assert [record["file_name"] for record in manifest] == ["a sound.wav", "b.wav", "c.wav"]
    assert references["clotho_eval_0001"] == [f"b.wav caption {i}." for i in range(1, 6)]
    assert "captions" not in manifest[0] and len(manifest[0]["audio_sha256"]) == 64
    assert check_manifest(manifest, references, 3, 5) == []


def test_check_manifest_detects_problems(clotho):
    manifest, references = build_eval_manifest(validate(clotho).records)
    manifest[1]["sample_id"] = manifest[0]["sample_id"]
    references["clotho_eval_0002"] = references["clotho_eval_0002"][:4]
    errors = check_manifest(manifest, references, 3, 5)
    assert "duplicate sample_id values" in errors
    assert any("without 5 references" in error for error in errors)
    assert any("14 reference captions" in error for error in errors)


def test_exclude_shared_sources():
    held_out = [{"file_name": "a.wav", "metadata": {"sound_id": "1"}}]
    candidates = [
        {"file_name": "a.wav", "metadata": {"sound_id": "9"}},
        {"file_name": "b.wav", "metadata": {"sound_id": "1"}},
        {"file_name": "c.wav", "metadata": {"sound_id": "2"}},
    ]
    kept, excluded = exclude_shared_sources(candidates, held_out)
    assert [r["file_name"] for r in kept] == ["c.wav"]
    assert [r["file_name"] for r in excluded] == ["a.wav", "b.wav"]


def test_exclude_shared_sources_ignores_unknown_placeholder():
    held_out = [{"file_name": "a.wav", "metadata": {"sound_id": "Not found"}}]
    candidates = [
        {"file_name": "b.wav", "metadata": {"sound_id": "Not found"}},
        {"file_name": "c.wav", "metadata": {"sound_id": ""}},
        {"file_name": "a.wav", "metadata": {"sound_id": "Not found"}},
    ]
    kept, excluded = exclude_shared_sources(candidates, held_out)
    assert [r["file_name"] for r in kept] == ["b.wav", "c.wav"]
    assert [r["file_name"] for r in excluded] == ["a.wav"]


def test_select_pilot_is_deterministic():
    names = [f"file_{i:03d}.wav" for i in range(100)]
    chosen = select_pilot(names, 20, 42)
    assert chosen == select_pilot(reversed(names), 20, 42)
    assert chosen != select_pilot(names, 20, 43)
    assert len(set(chosen)) == 20 and chosen == sorted(chosen)


def test_build_pilot_manifest(clotho):
    records = [record for record in validate(clotho, "validation", None).records if record["file_name"] != "v3.wav"]
    pilot = build_pilot_manifest(records)
    assert [record["sample_id"] for record in pilot] == ["clotho_pilot_00", "clotho_pilot_01", "clotho_pilot_02"]
    assert pilot[0]["split"] == "validation" and len(pilot[0]["captions"]) == 5


def test_checksums_roundtrip(tmp_path):
    files = [tmp_path / "a.json", tmp_path / "b.jsonl"]
    for path in files:
        path.write_text(path.name, encoding="utf-8")
    write_checksums(files, tmp_path / "SHA256SUMS")
    assert verify_checksums(tmp_path / "SHA256SUMS") == []
    files[1].write_text("changed", encoding="utf-8")
    assert verify_checksums(tmp_path / "SHA256SUMS") == ["b.jsonl: checksum changed"]


def run_prepare(clotho, tmp_path, *extra):
    manifests = tmp_path / "manifests"
    overrides = {
        "root": clotho,
        "evaluation.expected_recordings": 3,
        "evaluation.manifest": manifests / "eval.jsonl",
        "evaluation.references": manifests / "refs.json",
        "pilot.num_recordings": 2,
        "pilot.manifest": manifests / "pilot.jsonl",
    }
    args = [arg for key, value in overrides.items() for arg in ("--set", f"{key}={value}")]
    command = [sys.executable, "scripts/prepare_dataset.py", "--no-log-file", "--report-dir", str(tmp_path / "reports")]
    return subprocess.run([*command, *args, *extra], cwd=REPO_ROOT, capture_output=True, text=True), manifests


def test_prepare_dataset_end_to_end(clotho, tmp_path):
    result, manifests = run_prepare(clotho, tmp_path)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in manifests.iterdir()) == [
        "CLOTHO_LICENSE",
        "SHA256SUMS",
        "eval.jsonl",
        "pilot.jsonl",
        "refs.json",
    ]
    assert len(json.loads((manifests / "refs.json").read_text(encoding="utf-8"))) == 3
    assert len((manifests / "pilot.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    for name in ("dataset_validation.json", "dataset_validation.md", "manifest_report.json", "manifest_report.md"):
        assert (tmp_path / "reports" / name).is_file()
    assert "caption 1." not in (tmp_path / "reports" / "dataset_validation.json").read_text(encoding="utf-8")

    rerun, _ = run_prepare(clotho, tmp_path)
    assert rerun.returncode == 1 and "use --force" in rerun.stderr
    first = (manifests / "SHA256SUMS").read_text(encoding="utf-8")
    forced, _ = run_prepare(clotho, tmp_path, "--force")
    assert forced.returncode == 0, forced.stderr
    assert (manifests / "SHA256SUMS").read_text(encoding="utf-8") == first

    assert run_prepare(clotho, tmp_path, "--check")[0].returncode == 0
    (manifests / "refs.json").write_text("{}", encoding="utf-8")
    assert run_prepare(clotho, tmp_path, "--check")[0].returncode == 1


def test_prepare_dataset_fails_without_writing(clotho, tmp_path):
    (clotho / "audio" / "evaluation" / "b.wav").unlink()
    result, manifests = run_prepare(clotho, tmp_path)
    assert result.returncode == 1 and "no manifests written" in result.stderr
    assert not manifests.exists()
    assert (tmp_path / "reports" / "dataset_validation.json").is_file()


def test_prepare_dataset_excludes_shared_sources_from_pilot(clotho, tmp_path):
    path = clotho / "metadata" / "clotho_metadata_validation.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("validation-1", "evaluation-0"), encoding="utf-8")
    write_zenodo_record(clotho)
    result, manifests = run_prepare(clotho, tmp_path, "--set", "pilot.num_recordings=3")
    assert result.returncode == 0, result.stderr
    pilot = [json.loads(line) for line in (manifests / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["file_name"] for record in pilot] == ["v1.wav", "v3.wav", "v4.wav"]
    report = json.loads((tmp_path / "reports" / "dataset_validation.json").read_text(encoding="utf-8"))
    assert report["pilot"]["pool"] == 3
    assert report["pilot"]["excluded_shared_with_evaluation"] == [{"file_name": "v2.wav", "sound_id": "evaluation-0"}]
