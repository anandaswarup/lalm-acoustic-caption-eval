"""E02-E03 on a small synthetic Clotho-like fixture (CI has no copy of Clotho)."""

import json
import subprocess
import sys

from src.data.clotho import check_audio, load_split, validate_split
from src.data.manifest import (
    build_eval_manifest,
    check_manifest,
    select_sample,
    verify_checksums,
    write_checksums,
)
from src.utils.paths import REPO_ROOT
from tests.helpers import write_wav, write_zenodo_record

SOURCE = "https://zenodo.org/records/4783391"


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


def test_select_sample_is_deterministic():
    names = [f"file_{i:03d}.wav" for i in range(100)]
    chosen = select_sample(names, 20, 42)
    assert chosen == select_sample(reversed(names), 20, 42)
    assert chosen != select_sample(names, 20, 43)
    assert len(set(chosen)) == 20 and chosen == sorted(chosen)


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
        "refs.json",
    ]
    assert len(json.loads((manifests / "refs.json").read_text(encoding="utf-8"))) == 3
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
