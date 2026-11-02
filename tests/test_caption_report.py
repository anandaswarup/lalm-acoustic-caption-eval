"""E06-E09: completion report over a synthetic caption run."""

import json

import pytest

from src.data.manifest import write_jsonl
from src.generation.report import completion_report, is_wrapped_sentence, normalize_usage, report_markdown
from src.generation.runner import prompt_sha256

PROMPT = "Describe the sounds."


def record(sample_id, caption="A dog barks twice.", status="ok", prompt=PROMPT, **extra):
    return {
        "sample_id": sample_id,
        "status": status,
        "caption": caption,
        "model_version": "m-1",
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "finish_reason": "stop",
        "prompt_sha256": prompt_sha256(prompt),
        "latency_s": 1.0,
        "attempts": 1,
        "format_rejected": [],
        **extra,
    }


@pytest.fixture
def run(tmp_path):
    manifest = write_jsonl([{"sample_id": f"s{index}"} for index in range(6)], tmp_path / "manifest.jsonl")
    records = [
        record("s0", caption="old caption", prompt="an older prompt"),
        record("s0"),
        record("s1", caption='{"A dog barks."}', format_rejected=[{"caption": "{}"}, {"caption": "{}"}]),
        record("s2", format_rejected=[{"caption": '{"a": 1}'}]),
        record("s3", status="error", caption=None, error="boom", attempts=6),
        record("s4", caption="Rain falls.", prompt="an older prompt"),
        record("other", caption="not in the manifest"),
    ]
    output = write_jsonl(records, tmp_path / "model_captions.jsonl")
    return output, manifest


def test_completion_report_counts(run):
    output, manifest = run
    report = completion_report(output, manifest, PROMPT)
    assert report["complete"] is False
    assert report["statuses"] == {"error": 1, "ok": 4}
    assert report["problems"]["missing"] == ["s5"]
    assert report["problems"]["not_ok"] == [{"sample_id": "s3", "status": "error", "error": "boom"}]
    assert report["problems"]["other_prompt"] == ["s4"]
    assert report["records"] == {"lines": 7, "samples_with_record": 5, "not_in_manifest": 1}
    assert report["format"]["retried"] == 2
    assert report["format"]["fixed_by_retry"] == 1
    assert report["format"]["rejected_responses"] == 3
    assert report["format"]["json_like"] == report["format"]["json_like_unwrappable"] == ["s1"]
    assert report["transient_retries"] == {"samples": 1, "extra_attempts": 5}
    assert report["tokens"]["input"] == 400 and report["tokens"]["input_audio"] is None


def test_complete_run(tmp_path):
    manifest = write_jsonl([{"sample_id": "s0"}, {"sample_id": "s1"}], tmp_path / "manifest.jsonl")
    output = write_jsonl([record("s0"), record("s1")], tmp_path / "model_captions.jsonl")
    meta = {"prompt": {"version": "v1", "text": PROMPT}, "run_info": {"git_commit": "abc"}, "manifest": {}}
    report = completion_report(output, manifest, PROMPT, meta)
    assert report["complete"] is True
    assert report["last_invocation"]["prompt_matches"] is True
    assert report["last_invocation"]["manifest_sha256_matches"] is False


def test_markdown_has_no_caption_text(run):
    output, manifest = run
    text = report_markdown(completion_report(output, manifest, PROMPT), "Report")
    assert "s1" in text and "s5" in text
    for caption in ("dog barks", "Rain falls", "old caption", "not in the manifest"):
        assert caption not in text
    json.dumps(completion_report(output, manifest, PROMPT))  # serializable


@pytest.mark.parametrize(
    "caption, expected",
    [
        ('{"A dog barks."}', True),
        ('  {"A dog barks."}\n', True),
        ('{"sound": "dog"}', False),
        ('["A dog barks."]', False),
        ("A dog barks.", False),
        ('{""}', False),
        (None, False),
    ],
)
def test_is_wrapped_sentence(caption, expected):
    assert is_wrapped_sentence(caption) is expected


def test_normalize_usage():
    openai = {
        "prompt_tokens": 347,
        "completion_tokens": 26,
        "prompt_tokens_details": {"audio_tokens": 274, "text_tokens": 73},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    gemini = {
        "prompt_token_count": 661,
        "candidates_token_count": 19,
        "thoughts_token_count": 287,
        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 63}, {"modality": "AUDIO", "token_count": 598}],
    }
    assert normalize_usage(openai) == {"input": 347, "input_audio": 274, "input_text": 73, "output": 26, "thinking": 0}
    expected = {"input": 661, "input_audio": 598, "input_text": 63, "output": 19, "thinking": 287}
    assert normalize_usage(gemini) == expected
    assert normalize_usage({"input_tokens": 5, "output_tokens": 2})["output"] == 2
    assert set(normalize_usage(None).values()) == {None}
