"""E10: caption consolidation, normalization and checksums on a synthetic four-model run."""

import json

import pytest

from src.data.manifest import verify_checksums, write_checksums, write_json, write_jsonl
from src.generation.consolidate import consolidate, malformed_type, normalize_caption, report_markdown
from src.generation.runner import prompt_sha256

PROMPT = "Describe the sounds."
MODELS = ("m_a", "m_b", "m_c", "m_d")
IDS = [f"s{index}" for index in range(5)]


def record(sample_id, model_key, caption="A dog barks.", status="ok", prompt=PROMPT):
    return {
        "sample_id": sample_id,
        "model_key": model_key,
        "status": status,
        "caption": caption,
        "model_id": f"{model_key}-id",
        "model_version": "v1",
        "prompt_sha256": prompt_sha256(prompt),
    }


@pytest.fixture
def setup(tmp_path):
    manifest = write_jsonl([{"sample_id": sample_id} for sample_id in IDS], tmp_path / "manifest.jsonl")
    references = write_json({sample_id: [f"ref {i}" for i in range(5)] for sample_id in IDS}, tmp_path / "refs.json")
    special = {
        "s0": ' {"A dog barks loudly."}\n',
        "s1": '{"main_sound_sources": "dog", "audible_events": "barking"}',
        "s2": '{"result": "A dog barks."}',
        "s3": '{"error": "I cannot hear the audio."}',
    }

    def write(model_key, overrides=None, extra=()):
        records = [record(sample_id, model_key) for sample_id in IDS]
        if model_key == "m_a":
            records = [record("s0", model_key, caption="superseded")] + [
                record(sample_id, model_key, caption=special.get(sample_id, "A dog barks.")) for sample_id in IDS
            ]
        records = [(overrides or {}).get(item["sample_id"], item) for item in records] + list(extra)
        return write_jsonl(records, tmp_path / f"{model_key}_captions.jsonl")

    sources = {model_key: write(model_key) for model_key in MODELS}
    return {"sources": sources, "manifest": manifest, "references": references, "write": write, "tmp": tmp_path}


def run(setup, sources=None):
    return consolidate(sources or setup["sources"], setup["manifest"], setup["references"], PROMPT)


def test_consolidate_records_and_report(setup):
    records, report, errors = run(setup)
    assert errors == []
    assert len(records) == 20 and report["captions"] == 20 and report["references"] == 25
    assert [(r["sample_id"], r["model_key"]) for r in records[:5]] == [
        ("s0", "m_a"), ("s0", "m_b"), ("s0", "m_c"), ("s0", "m_d"), ("s1", "m_a"),
    ]  # fmt: skip
    first = records[0]
    assert first["caption"] == "A dog barks loudly." and first["normalization"] == "unwrapped"
    assert first["caption_raw"] == ' {"A dog barks loudly."}\n'
    per_model = report["per_model"]["m_a"]
    assert per_model["normalization"] == {"none": 1, "unwrapped": 1, "malformed": 3}
    assert per_model["malformed_types"] == {"fields": 1, "refusal": 1, "result": 1}
    assert per_model["malformed_ids"] == {"s1": "fields", "s2": "result", "s3": "refusal"}
    assert per_model["record_lines"] == 6
    assert report["per_model"]["m_b"]["normalization"] == {"none": 5, "unwrapped": 0, "malformed": 0}
    assert report["valid_for_all_models"] == {"count": 2, "sample_ids": ["s0", "s4"]}
    malformed = [r for r in records if r["normalization"] == "malformed"]
    assert all(r["caption"] == r["caption_raw"].strip() for r in malformed)  # scored as returned


@pytest.mark.parametrize(
    "overrides, extra, message",
    [
        ({"s2": None}, (), "1 samples missing"),
        ({}, (record("s9", "m_b"),), "not in the manifest"),
        ({"s2": record("s2", "m_b", status="truncated")}, (), "status truncated"),
        ({"s2": record("s2", "m_b", caption="  ")}, (), "empty caption"),
        ({"s2": record("s2", "m_b", prompt="old")}, (), "another prompt"),
        ({"s2": record("s2", "m_c")}, (), "record is for model m_c"),
    ],
)
def test_consolidate_errors(setup, overrides, extra, message):
    path = setup["write"]("m_b", overrides, extra)
    path.write_text("".join(line + "\n" for line in path.read_text().splitlines() if line != "null"))
    records, report, errors = run(setup, {**setup["sources"], "m_b": path})
    assert records == [] and any(message in error for error in errors), errors
    assert "not frozen" in report_markdown(report, {})


def test_duplicate_sample_keeps_last_record(setup):
    path = setup["write"]("m_b", extra=(record("s1", "m_b", caption="Rain falls."),))
    records, _, errors = run(setup, {**setup["sources"], "m_b": path})
    assert errors == []
    assert next(r for r in records if (r["sample_id"], r["model_key"]) == ("s1", "m_b"))["caption"] == "Rain falls."


def test_references_must_align(setup):
    refs = {sample_id: ["r"] * 5 for sample_id in IDS[:-1]} | {"s4": ["r"] * 4}
    write_json(refs, setup["references"])
    assert any("without 5 captions" in error for error in run(setup)[2])


def test_markdown_has_no_caption_text(setup):
    records, report, _ = run(setup)
    text = report_markdown(report, {"all_model_captions.jsonl": "abc"})
    assert "`s1` | fields" in text and "`all_model_captions.jsonl`" in text
    for phrase in ("dog", "barks", "cannot hear"):
        assert phrase not in text
    json.dumps(report)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  A dog barks. ", ("A dog barks.", "none", None)),
        ('{"A dog barks."}', ("A dog barks.", "unwrapped", None)),
        ('{"a": "x", "b": "y"}', ('{"a": "x", "b": "y"}', "malformed", "fields")),
        ('{"result": "x"}', ('{"result": "x"}', "malformed", "result")),
        ('{"error": "x"}', ('{"error": "x"}', "malformed", "refusal")),
        ('{"caption": "x"}', ('{"caption": "x"}', "malformed", "other")),
        ('["x"]', ('["x"]', "malformed", "other")),
        ("{not json", ("{not json", "malformed", "other")),
    ],
)
def test_normalize_caption(raw, expected):
    result = normalize_caption(raw)
    assert (result["caption"], result["normalization"], result["malformed_type"]) == expected


def test_malformed_type_other_for_scalars():
    assert malformed_type("[1, 2]") == "other" and malformed_type('"x"') == "other"


def test_checksums_with_subdirectories(tmp_path):
    (tmp_path / "sub").mkdir()
    files = [tmp_path / "a.jsonl", tmp_path / "sub" / "b.jsonl"]
    for path in files:
        path.write_text("{}\n")
    write_checksums(files, tmp_path / "SHA256SUMS")
    names = [line.split()[1] for line in (tmp_path / "SHA256SUMS").read_text().splitlines()]
    assert names == ["a.jsonl", "sub/b.jsonl"]
    assert verify_checksums(tmp_path / "SHA256SUMS") == []
    files[1].write_text("[]\n")
    assert verify_checksums(tmp_path / "SHA256SUMS") == ["sub/b.jsonl: checksum changed"]
