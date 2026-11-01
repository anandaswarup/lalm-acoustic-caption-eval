"""E04: adapters, resumable runner, generation script and HF Jobs launcher, without network or GPUs."""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from src.data.clotho import validate_split
from src.data.manifest import build_eval_manifest, write_jsonl
from src.generation.runner import latest_records, load_samples, run_generation
from src.models.audio import load_mono_float, read_wav_bytes
from src.models.base import CaptionRequest, TransientError, classify, load_adapter, merge_decoding
from src.models.gemini import GeminiAdapter
from src.models.mock import MockAdapter
from src.models.openai_audio import OpenAIAudioAdapter
from src.utils.paths import REPO_ROOT

SOURCE = "https://zenodo.org/records/4783391"
GENERATION = {
    "prompt": {"version": "v1", "text": "Describe the sounds."},
    "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 2048},
    "requests": {"timeout_seconds": 5, "max_retries": 2, "backoff_seconds": 0.0},
}
NO_SLEEP = lambda seconds: None  # noqa: E731


@pytest.fixture
def manifest(clotho, tmp_path):
    records, _ = build_eval_manifest(validate_split(clotho, "evaluation", 3, 5, SOURCE).records)
    return write_jsonl(records, tmp_path / "eval.jsonl")


@pytest.fixture
def samples(manifest, clotho):
    return load_samples(manifest, clotho)


def mock_adapter(behaviour=None):
    return load_adapter("mock_model", {"adapter": "mock", "model_id": "mock", "mock": behaviour or {}}, GENERATION)


def generate(adapter, samples, output, resume=True):
    return run_generation(
        adapter, samples, output, "Describe.", GENERATION["decoding"], GENERATION["requests"], {}, resume, NO_SLEEP
    )


# --- base -------------------------------------------------------------------


def test_classify():
    assert classify("A dog barks.", truncated=False) == "ok"
    assert classify("  ", truncated=False) == "empty"
    assert classify(None, truncated=False) == "empty"
    assert classify("A dog", truncated=True) == "truncated"


def test_merge_decoding_applies_overrides():
    merged = merge_decoding(GENERATION, {"decoding_overrides": {"seed": 42, "temperature": 0.5}})
    assert merged == {"temperature": 0.5, "top_p": 1.0, "max_output_tokens": 2048, "seed": 42}
    assert merge_decoding(GENERATION, {"decoding_overrides": None}) == GENERATION["decoding"]


def test_load_adapter_rejects_unknown():
    with pytest.raises(ValueError, match="unknown adapter"):
        load_adapter("x", {"adapter": "nope", "model_id": "x"}, GENERATION)


# --- runner -----------------------------------------------------------------


def test_load_samples_selection(manifest, clotho):
    assert [s["sample_id"] for s in load_samples(manifest, clotho, limit=2)] == ["clotho_eval_0000", "clotho_eval_0001"]
    chosen = load_samples(manifest, clotho, sample_ids=["clotho_eval_0002"])
    assert [s["sample_id"] for s in chosen] == ["clotho_eval_0002"]
    assert chosen[0]["resolved_audio_path"] == clotho / "audio/evaluation/c.wav"
    with pytest.raises(ValueError, match="not in manifest"):
        load_samples(manifest, clotho, sample_ids=["clotho_eval_9999"])


def test_run_generation_writes_records_and_meta(samples, tmp_path):
    output = tmp_path / "out" / "mock_captions.jsonl"
    summary = generate(mock_adapter(), samples, output)
    assert summary["statuses"] == {"ok": 3} and summary["missing"] == [] and summary["failed"] == []
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["sample_id"] for r in records] == ["clotho_eval_0000", "clotho_eval_0001", "clotho_eval_0002"]
    first = records[0]
    assert first["caption"] == "Mock caption for clotho_eval_0000." and first["attempts"] == 1
    assert first["model_key"] == "mock_model" and first["latency_s"] >= 0 and first["timestamp_utc"]
    assert first["request_params"]["max_output_tokens"] == 2048
    meta = json.loads((tmp_path / "out" / "mock_captions.meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished" and meta["summary"]["statuses"] == {"ok": 3}
    assert meta["adapter"]["adapter"] == "MockAdapter"


def test_run_generation_resumes(samples, tmp_path):
    output = tmp_path / "captions.jsonl"
    generate(mock_adapter({"clotho_eval_0001": "error"}), samples, output)
    adapter = mock_adapter()
    summary = generate(adapter, samples, output)
    assert adapter.calls == {"clotho_eval_0001": 1}  # only the failed sample is retried
    assert summary["statuses"] == {"ok": 3}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4  # earlier attempts are kept
    assert latest_records(output)["clotho_eval_0001"]["status"] == "ok"


def test_resume_ignores_records_from_another_prompt(samples, tmp_path):
    output = tmp_path / "captions.jsonl"
    generate(mock_adapter(), samples, output)
    assert latest_records(output)["clotho_eval_0000"]["prompt_sha256"] == hashlib.sha256(b"Describe.").hexdigest()
    adapter = mock_adapter()
    run_generation(adapter, samples, output, "Describe again.", GENERATION["decoding"], GENERATION["requests"], {})
    assert sum(adapter.calls.values()) == 3  # a changed prompt means nothing is done yet
    adapter = mock_adapter()
    run_generation(adapter, samples, output, "Describe again.", GENERATION["decoding"], GENERATION["requests"], {})
    assert adapter.calls == {}


class SlowMockAdapter(MockAdapter):
    """Records how many requests overlap."""

    def __init__(self, *args):
        super().__init__(*args)
        self.lock, self.active, self.max_active = threading.Lock(), 0, 0

    def caption(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return super().caption(request)


def test_concurrent_generation_matches_sequential(samples, tmp_path):
    config = {"adapter": "mock", "model_id": "mock", "mock": {"clotho_eval_0001": "transient:2"}}
    requests_config = {**GENERATION["requests"], "concurrency": 3}
    adapter = SlowMockAdapter("mock_model", config, GENERATION)
    output = tmp_path / "concurrent.jsonl"
    summary = run_generation(
        adapter, samples, output, "Describe.", GENERATION["decoding"], requests_config, {}, sleep=NO_SLEEP
    )
    assert adapter.max_active > 1 and summary["statuses"] == {"ok": 3}
    assert adapter.calls == {"clotho_eval_0000": 1, "clotho_eval_0001": 3, "clotho_eval_0002": 1}
    sequential = tmp_path / "sequential.jsonl"
    generate(mock_adapter(), samples, sequential)
    captions = {key: record["caption"] for key, record in latest_records(output).items()}
    assert captions == {key: record["caption"] for key, record in latest_records(sequential).items()}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3  # one complete line per sample
    resumed = SlowMockAdapter("mock_model", config, GENERATION)
    run_generation(resumed, samples, output, "Describe.", GENERATION["decoding"], requests_config, {})
    assert resumed.calls == {}


def test_json_output_is_requested_again(samples, tmp_path):
    output = tmp_path / "captions.jsonl"
    requests_config = {**GENERATION["requests"], "format_attempts": 3}
    behaviour = {"clotho_eval_0000": "json:1", "clotho_eval_0001": "json:5"}
    adapter = mock_adapter(behaviour)
    run_generation(adapter, samples, output, "Describe.", GENERATION["decoding"], requests_config, {}, sleep=NO_SLEEP)
    latest = latest_records(output)
    fixed, stuck, clean = (latest[f"clotho_eval_000{i}"] for i in range(3))
    assert adapter.calls == {"clotho_eval_0000": 2, "clotho_eval_0001": 3, "clotho_eval_0002": 1}
    assert fixed["caption"] == "Mock caption for clotho_eval_0000." and len(fixed["format_rejected"]) == 1
    assert fixed["format_rejected"][0]["caption"].startswith('{"main_sound_sources"')
    assert stuck["status"] == "ok" and stuck["caption"].startswith("{") and len(stuck["format_rejected"]) == 2
    assert clean["format_rejected"] == []
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3  # retries live inside one record


def test_run_generation_no_resume_recaptions_everything(samples, tmp_path):
    output = tmp_path / "captions.jsonl"
    generate(mock_adapter(), samples, output)
    adapter = mock_adapter()
    generate(adapter, samples, output, resume=False)
    assert sum(adapter.calls.values()) == 3


def test_transient_errors_are_retried(samples, tmp_path):
    adapter = mock_adapter({"clotho_eval_0000": "transient:2", "clotho_eval_0001": "transient:9"})
    summary = generate(adapter, samples, tmp_path / "captions.jsonl")
    latest = latest_records(tmp_path / "captions.jsonl")
    assert latest["clotho_eval_0000"]["status"] == "ok" and latest["clotho_eval_0000"]["attempts"] == 3
    assert latest["clotho_eval_0001"]["status"] == "error" and latest["clotho_eval_0001"]["attempts"] == 3
    assert "TransientError after 3 attempts" in latest["clotho_eval_0001"]["error"]
    assert summary["failed"] == ["clotho_eval_0001"]


def test_permanent_empty_and_truncated(samples, tmp_path):
    adapter = mock_adapter({"clotho_eval_0000": "error", "clotho_eval_0001": "empty", "clotho_eval_0002": "truncated"})
    summary = generate(adapter, samples, tmp_path / "captions.jsonl")
    assert summary["statuses"] == {"empty": 1, "error": 1, "truncated": 1}
    error = latest_records(tmp_path / "captions.jsonl")["clotho_eval_0000"]["error"]
    assert error == "ValueError: simulated permanent failure"


def test_bad_audio_is_never_sent(samples, clotho, tmp_path):
    write_wav_noise(clotho / "audio/evaluation/a sound.wav")  # clotho_eval_0000: checksum changes
    (clotho / "audio/evaluation/b.wav").unlink()  # clotho_eval_0001: missing
    adapter = mock_adapter()
    generate(adapter, samples, tmp_path / "captions.jsonl")
    latest = latest_records(tmp_path / "captions.jsonl")
    assert adapter.calls == {"clotho_eval_0002": 1}
    assert "SHA-256 does not match" in latest["clotho_eval_0000"]["error"]
    assert "not found" in latest["clotho_eval_0001"]["error"]


def write_wav_noise(path, rate=44100, seconds=0.2, channels=1):
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, (int(rate * seconds), channels)).astype(np.float32)
    sf.write(path, noise, rate, subtype="PCM_16")


# --- generate_captions.py -----------------------------------------------------


def run_script(clotho, manifest, tmp_path, *extra):
    command = [
        sys.executable, "scripts/generate_captions.py", "--no-log-file", "--model", "gpt_audio_1_5",
        "--set", "models.gpt_audio_1_5.adapter=mock", "--set", f"runs.evaluation.manifest={manifest}",
        "--output-dir", str(tmp_path / "out"), *extra,
    ]  # fmt: skip
    env = {**os.environ, "CLOTHO_ROOT": str(clotho)}
    return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def test_generate_script_end_to_end(clotho, manifest, tmp_path):
    result = run_script(clotho, manifest, tmp_path, "--limit", "2")
    assert result.returncode == 0, result.stderr
    output = tmp_path / "out" / "gpt_audio_1_5_captions.jsonl"
    assert list(latest_records(output)) == ["clotho_eval_0000", "clotho_eval_0001"]
    meta = json.loads((tmp_path / "out" / "gpt_audio_1_5_captions.meta.json").read_text(encoding="utf-8"))
    assert meta["run"] == "evaluation" and meta["decoding"]["seed"] == 42 and len(meta["manifest"]["sha256"]) == 64
    assert meta["selection"] == {"limit": 2, "sample_ids": None, "resume": True}


def test_generate_script_fails_on_errors(clotho, manifest, tmp_path):
    result = run_script(clotho, manifest, tmp_path, "--set", "models.gpt_audio_1_5.mock={clotho_eval_0002: error}")
    assert result.returncode == 1 and "not ok" in result.stderr


def test_generate_script_rejects_consolidate(clotho, manifest, tmp_path):
    result = run_script(clotho, manifest, tmp_path, "--run", "consolidate")
    assert result.returncode == 1 and "E10" in result.stderr


# --- audio --------------------------------------------------------------------


def test_read_wav_bytes_is_unchanged(tmp_path):
    path = tmp_path / "a.wav"
    write_wav_noise(path)
    data, details = read_wav_bytes(path)
    assert data == path.read_bytes() and details["sample_rate"] == 44100 and details["channels"] == 1


def test_load_mono_float_resamples_without_cropping(tmp_path):
    pytest.importorskip("soxr")
    path = tmp_path / "stereo.wav"
    write_wav_noise(path, seconds=1.0, channels=2)
    audio, rate, details = load_mono_float(path, 16000)
    assert rate == 16000 and audio.dtype == np.float32 and audio.ndim == 1
    assert abs(audio.shape[0] - 16000) <= 1
    assert details["downmix"] == "mean of channels" and details["resample"]["from"] == 44100
    _, same_rate, unchanged = load_mono_float(path, 44100)
    assert same_rate == 44100 and "resample" not in unchanged


# --- OpenAI adapter (fake client) --------------------------------------------


class FakeError(Exception):
    def __init__(self, message="", param=None, code=None):
        super().__init__(message)
        self.param, self.code = param, code


def fake_openai_module():
    names = ["APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError", "BadRequestError"]
    return SimpleNamespace(**{name: type(name, (FakeError,), {}) for name in names}, __version__="fake")


def fake_openai_response(content="A dog barks.", finish_reason="stop"):
    message = SimpleNamespace(content=content)
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
        model="gpt-audio-1.5-2026-01-01",
        system_fingerprint="fp_1",
        model_dump=lambda mode=None: {"id": "resp"},
    )


class FakeOpenAIClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def openai_adapter(outcomes):
    config = {"adapter": "openai_audio", "model_id": "gpt-audio-1.5", "api_key_env": "OPENAI_API_KEY"}
    adapter = OpenAIAudioAdapter("gpt_audio_1_5", config, GENERATION)
    adapter.openai, adapter.client, adapter.dropped = fake_openai_module(), FakeOpenAIClient(outcomes), {}
    return adapter


def wav_request(tmp_path, decoding=None):
    path = tmp_path / "a.wav"
    write_wav_noise(path)
    return CaptionRequest("s1", path, "Describe.", decoding or {**GENERATION["decoding"], "seed": 42})


def test_openai_request_format(tmp_path):
    adapter = openai_adapter([fake_openai_response()])
    request = wav_request(tmp_path)
    result = adapter.caption(request)
    call = adapter.client.calls[0]
    assert call["model"] == "gpt-audio-1.5" and call["modalities"] == ["text"] and call["store"] is False
    assert (call["temperature"], call["top_p"], call["seed"], call["max_completion_tokens"]) == (0.0, 1.0, 42, 2048)
    audio, text = call["messages"][0]["content"]
    assert text == {"type": "text", "text": "Describe."}
    assert audio["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio["input_audio"]["data"]) == request.audio_path.read_bytes()
    assert result.status == "ok" and result.caption == "A dog barks." and result.model_version.startswith("gpt-audio")
    assert result.usage["system_fingerprint"] == "fp_1" and result.deviations == []


def test_openai_drops_rejected_parameter(tmp_path):
    module = fake_openai_module()
    rejected = module.BadRequestError("Unsupported parameter: 'seed' is not supported with this model.", param="seed")
    adapter = openai_adapter([rejected, fake_openai_response(), fake_openai_response()])
    adapter.openai = module
    first = adapter.caption(wav_request(tmp_path))
    assert "seed" not in adapter.client.calls[1] and first.deviations == ["parameter 'seed' rejected by API, not sent"]
    adapter.caption(wav_request(tmp_path))
    assert "seed" not in adapter.client.calls[2]  # stays dropped for the rest of the run


def test_openai_transient_and_truncated(tmp_path):
    module = fake_openai_module()
    adapter = openai_adapter([module.RateLimitError("429")])
    adapter.openai = module
    with pytest.raises(TransientError):
        adapter.caption(wav_request(tmp_path))
    truncated = openai_adapter([fake_openai_response("A dog", "length")]).caption(wav_request(tmp_path))
    assert truncated.status == "truncated"


def test_openai_other_bad_request_is_raised(tmp_path):
    module = fake_openai_module()
    adapter = openai_adapter([module.BadRequestError("Invalid audio file")])
    adapter.openai = module
    with pytest.raises(module.BadRequestError):
        adapter.caption(wav_request(tmp_path))


# --- Gemini adapter (fake client) --------------------------------------------


def fake_genai_types():
    return SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ThinkingConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Part=SimpleNamespace(from_bytes=lambda data, mime_type: {"data": data, "mime_type": mime_type}),
    )


def fake_gemini_response(text="Rain falls on a roof.", finish="STOP"):
    usage = SimpleNamespace(model_dump=lambda **kwargs: {"prompt_token_count": 700, "thoughts_token_count": 40})
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=usage,
        model_version="gemini-3.8-flash-001",
        model_dump=lambda **kwargs: {"id": "resp"},
    )


def gemini_adapter(outcomes):
    config = {"adapter": "gemini", "model_id": "gemini-3.8-flash", "api_key_env": "GEMINI_API_KEY"}
    adapter = GeminiAdapter("gemini_3_8_flash", config, GENERATION)
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    adapter.types, adapter.errors = fake_genai_types(), SimpleNamespace(APIError=FakeError)
    adapter.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return adapter, calls


def test_gemini_request_format(tmp_path):
    adapter, calls = gemini_adapter([fake_gemini_response()])
    request = wav_request(tmp_path, {**GENERATION["decoding"], "seed": 42, "thinking_level": "low"})
    result = adapter.caption(request)
    call = calls[0]
    assert call["model"] == "gemini-3.8-flash"
    audio, prompt = call["contents"]
    assert prompt == "Describe." and audio == {"data": request.audio_path.read_bytes(), "mime_type": "audio/wav"}
    config = call["config"]
    assert (config.temperature, config.top_p, config.seed, config.max_output_tokens) == (0.0, 1.0, 42, 2048)
    assert config.thinking_config.thinking_level == "low"
    assert result.status == "ok" and result.model_version == "gemini-3.8-flash-001"
    assert result.usage["thoughts_token_count"] == 40
    assert result.deviations == ["max_output_tokens includes thinking tokens"]


def test_gemini_transient_permanent_truncated(tmp_path):
    adapter, _ = gemini_adapter([FakeError("busy", code=503)])
    with pytest.raises(TransientError):
        adapter.caption(wav_request(tmp_path))
    adapter, _ = gemini_adapter([FakeError("bad", code=400)])
    with pytest.raises(FakeError):
        adapter.caption(wav_request(tmp_path))
    adapter, _ = gemini_adapter([fake_gemini_response("Rain", "MAX_TOKENS")])
    assert adapter.caption(wav_request(tmp_path)).status == "truncated"


# --- Audio Flamingo 3 adapter (fake processor and model) ---------------------


class FakeTokens:
    """The slice of the torch.Tensor API the HF adapters use."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def __getitem__(self, index):
        return FakeTokens(self.values[index])

    def __len__(self):
        return len(self.values)

    def tolist(self):
        return self.values.tolist()

    def unsqueeze(self, dim):
        return FakeTokens(np.expand_dims(self.values, dim))


class FakeInputs(dict):
    def __init__(self, **tensors):
        super().__init__(**tensors)
        self.moves = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self


class FakeAF3Processor:
    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(eos_token_id=2)

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        self.inputs = FakeInputs(input_ids=FakeTokens([[5, 6, 7]]))
        return self.inputs

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [" A fan whirs. "]


def test_audio_flamingo_passes_resampled_array(tmp_path):
    pytest.importorskip("soxr")
    from contextlib import nullcontext

    from src.models.hf_audio_flamingo import AudioFlamingo3Adapter

    config = {"adapter": "audio_flamingo_3", "model_id": "nvidia/audio-flamingo-3-hf", "revision": "abc"}
    adapter = AudioFlamingo3Adapter("audio_flamingo_3", config, GENERATION)
    adapter.revision, adapter.processor, adapter.dtype = "abc", FakeAF3Processor(), "bf16"
    adapter.torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: False))
    adapter.model = SimpleNamespace(
        device="cpu",
        generation_config=SimpleNamespace(eos_token_id=2),
        generate=lambda **kwargs: FakeTokens([[5, 6, 7, 11, 2]]),
    )
    result = adapter.caption(wav_request(tmp_path))

    conversation, kwargs = adapter.processor.calls[0]
    audio_item, text_item = conversation[0]["content"]
    assert text_item == {"type": "text", "text": "Describe."}
    assert audio_item["type"] == "audio" and "path" not in audio_item
    assert isinstance(audio_item["audio"], np.ndarray) and audio_item["audio"].dtype == np.float32
    assert abs(audio_item["audio"].shape[0] - 3200) <= 1  # 0.2 s at 44.1 kHz, resampled once by soxr
    assert kwargs["sampling_rate"] == 16000 and kwargs["tokenize"] and kwargs["return_dict"]
    assert adapter.processor.inputs.moves == [(("cpu",), {"dtype": "bf16"})]
    assert result.status == "ok" and result.caption == "A fan whirs." and result.finish_reason == "stop"
    assert result.usage["input_tokens"] == 3 and result.usage["output_tokens"] == 2
    assert result.audio_preprocessing["resample"]["from"] == 44100


class FakeQwenProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id=2)

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs["tokenize"] is False
        return "<prompt>"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.inputs = FakeInputs(input_ids=FakeTokens([[5, 6, 7]]))
        return self.inputs

    def batch_decode(self, tokens, **kwargs):
        return ["A fan whirs."]


def test_qwen_calls_thinker_with_shared_decoding(tmp_path):
    pytest.importorskip("soxr")
    from contextlib import nullcontext

    from src.models.hf_qwen_omni import Qwen25OmniAdapter

    calls = []

    def thinker_generate(**kwargs):
        calls.append(kwargs)
        return FakeTokens([[5, 6, 7, 11, 2]])

    def combined_generate(**kwargs):
        raise AssertionError("the combined generate() must not be used")

    config = {"adapter": "qwen2_5_omni", "model_id": "Qwen/Qwen2.5-Omni-7B", "revision": "abc"}
    adapter = Qwen25OmniAdapter("qwen2_5_omni", config, GENERATION)
    adapter.revision, adapter.processor, adapter.dtype = "abc", FakeQwenProcessor(), "bf16"
    adapter.torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: False))
    adapter.model = SimpleNamespace(
        device="cpu",
        generation_config=SimpleNamespace(eos_token_id=2),
        generate=combined_generate,
        thinker=SimpleNamespace(generate=thinker_generate),
    )
    result = adapter.caption(wav_request(tmp_path, {**GENERATION["decoding"], "max_output_tokens": 2048}))

    assert calls[0]["max_new_tokens"] == 2048 and calls[0]["do_sample"] is False
    assert "return_audio" not in calls[0] and "thinker_max_new_tokens" not in calls[0]
    assert adapter.processor.kwargs["sampling_rate"] == 16000
    assert result.status == "ok" and result.request_params == {"max_new_tokens": 2048, "do_sample": False}


def test_error_records_keep_a_traceback(samples, tmp_path):
    sample_id = samples[0]["sample_id"]
    output = tmp_path / "out.jsonl"
    generate(mock_adapter({sample_id: "error"}), samples[:1], output)
    record = latest_records(output)[sample_id]
    assert record["status"] == "error" and "Traceback" in record["error_traceback"]
    assert "mock.py" in record["error_traceback"] and "in caption" in record["error_traceback"]


# --- hf_jobs.py -----------------------------------------------------------------


@pytest.fixture
def hf_jobs():
    spec = importlib.util.spec_from_file_location("hf_jobs", REPO_ROOT / "scripts" / "hf_jobs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SETTINGS = {"bucket": "user/bucket", "mount": "/mnt", "image": "astral/uv:tag", "timeout": "4h"}


def test_submit_command(hf_jobs):
    model = {"hf_jobs": {"flavor": "a100-large"}}
    command = hf_jobs.submit_command(SETTINGS, "qwen2_5_omni", model, "pilot", "abc123" * 7, ["--limit", "1"])
    assert command[:4] == ["hf", "jobs", "run", "--detach"]
    joined = " ".join(command[:-1])
    for expected in (
        "--flavor a100-large",
        "--timeout 4h",
        "--secrets HF_TOKEN",
        "CLOTHO_ROOT=/mnt/clotho",
        "--volume hf://buckets/user/bucket:/mnt",
        "--label model=qwen2_5_omni",
        "astral/uv:tag",
    ):
        assert expected in joined
    script = command[-1]
    assert command[-3:-1] == ["bash", "-c"] and f"cd /mnt/code/{'abc123' * 7}" in script
    assert "uv pip install --system --index-strategy unsafe-best-match -r requirements-hf.txt" in script
    assert "--output-dir /mnt/outputs/pilot --no-log-file --limit 1" in script


def test_submit_command_output_prefix(hf_jobs):
    model = {"hf_jobs": {"flavor": "a100-large"}}
    command = hf_jobs.submit_command(SETTINGS, "qwen2_5_omni", model, "pilot", "abc", [], prefix="smoke")
    assert "--output-dir /mnt/smoke/pilot" in command[-1] and "/mnt/outputs" not in command[-1]
    assert "prefix=smoke" in command


@pytest.mark.parametrize("value", ["clotho", "code", "a/b", "", ".."])
def test_output_prefix_rejects_reserved_and_nested(hf_jobs, value):
    with pytest.raises(argparse.ArgumentTypeError):
        hf_jobs.output_prefix(value)


def test_snapshot_paths_exclude_secrets_and_docs(hf_jobs):
    assert not any(path.startswith((".env", "docs", "outputs")) for path in hf_jobs.SNAPSHOT_PATHS)


def test_stage_audio_only_listed_files(hf_jobs, clotho, tmp_path):
    staging = tmp_path / "staging"
    hf_jobs.stage_audio(clotho, ["audio/evaluation/a sound.wav", "audio/validation/v2.wav"], staging)
    staged = sorted(str(path.relative_to(staging)) for path in staging.rglob("*.wav"))
    assert staged == ["audio/evaluation/a sound.wav", "audio/validation/v2.wav"]
    name = "audio/evaluation/a sound.wav"
    assert (staging / name).read_bytes() == (clotho / name).read_bytes()


def test_submit_refuses_uncommitted_changes(hf_jobs, monkeypatch):
    monkeypatch.setattr(hf_jobs, "git", lambda *args: " M src/x.py" if args[0] == "status" else "abc")
    monkeypatch.setattr(sys, "argv", ["hf_jobs.py", "--no-log-file", "submit", "--model", "qwen2_5_omni",
                                      "--run", "pilot", "--dry-run"])  # fmt: skip
    assert hf_jobs.main() == 1


def test_submit_refuses_local_models(hf_jobs, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hf_jobs.py", "--no-log-file", "submit", "--model", "gpt_audio_1_5",
                                      "--run", "pilot", "--dry-run"])  # fmt: skip
    assert hf_jobs.main() == 1
