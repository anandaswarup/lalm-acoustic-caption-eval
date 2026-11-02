# Caption completion report: Audio Flamingo 3 (evaluation)

Output: `outputs/captions/evaluation/audio_flamingo_3_captions.jsonl` (SHA-256 `eaa018aecf42b4f052ce94924004c9f44108dafd3178209bb441c61090796e0a`)

## Completeness

| Item | Value |
|---|---|
| Complete | **yes** |
| Manifest samples | 1045 |
| Samples with a record | 1045 |
| Record lines (all attempts) | 1045 |
| Statuses | ok: 1045 |
| Missing | none |
| Not ok | none |
| Ok with another prompt | none |
| Prompt SHA-256 | `6a50f6d3766d4d1bdfcbc16382d99f69e91158c2d1156cb9da03973293351cbb` |

## Format

| Item | Value |
|---|---|
| Format retries (samples) | 0 |
| Fixed by a retry | 0 |
| Rejected responses | 0 |
| Still JSON-like | none |
| Of which exactly {"…"} (E10 unwraps) | none |
| Multi-line captions | none |
| Finish reasons | stop: 1045 |
| Caption words | min: 4; mean: 18.96; median: 18; max: 44 |

## Usage

| Item | Value |
|---|---|
| Tokens (ok responses) | input: 670051; input_audio: n/a; input_text: n/a; output: 25496; thinking: n/a |
| Peak GPU memory (GB) | 16.72 |
| Latency (s) | min: 0.21; mean: 0.69; median: 0.669; max: 1.534; total: 722.024 |
| Samples with transient retries | 0 |
| Extra attempts | 0 |

## Model

| Item | Value |
|---|---|
| Model versions | 7d4bae64ee29878af6504ae6f6bb3e40492838ad: 1045 |
| System fingerprints | none |
| Deviations | none |

## Last invocation (meta file)

| Item | Value |
|---|---|
| Status | finished |
| Started utc | 2026-11-02T06:01:58+00:00 |
| Finished utc | 2026-11-02T06:22:12+00:00 |
| Git commit | bd4db4727659eacf0ec579990e88aa070bd60859 |
| Git dirty | False |
| Generation frozen | True |
| Manifest sha256 matches | True |
| Prompt version | v3 |
| Prompt matches | True |
| Decoding | temperature: 0.0; top_p: 1.0; max_output_tokens: 2048 |
| Requests | concurrency: 1; format_attempts: 3; timeout_seconds: 120; max_retries: 5; backoff_seconds: 2.0 |
| Adapter | adapter: AudioFlamingo3Adapter; model_id: nvidia/audio-flamingo-3-hf; revision: 7d4bae64ee29878af6504ae6f6bb3e40492838ad; library: transformers; library_version: 5.22.1; torch_version: 2.14.1+cu126; dtype: torch.bfloat16; cuda: 12.6; gpu: name: NVIDIA A100-SXM4-80GB; memory_gb: 85.1; count: 1; system_prompt: n/a; repetition_penalty: n/a |
