# Caption completion report: Gemini 3.8 Flash (evaluation)

Output: `outputs/captions/evaluation/gemini_3_8_flash_captions.jsonl` (SHA-256 `2c020d6b6121156f0b6c3d0c968f970a8f62220bd4d15eee123f310df50e2c93`)

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
| Of which exactly {"…"} (E09 unwraps) | none |
| Multi-line captions | none |
| Finish reasons | STOP: 1045 |
| Caption words | min: 8; mean: 16.36; median: 16; max: 36 |

## Usage

| Item | Value |
|---|---|
| Tokens (ok responses) | input: 651751; input_audio: 585916; input_text: 65835; output: 21093; thinking: 39064 |
| Peak GPU memory (GB) | n/a |
| Latency (s) | min: 3.086; mean: 6.58; median: 5.831; max: 68.116; total: 6875.636 |
| Samples with transient retries | 0 |
| Extra attempts | 0 |

## Model

| Item | Value |
|---|---|
| Model versions | gemini-3.8-flash: 1045 |
| System fingerprints | none |
| Deviations | max_output_tokens includes thinking tokens: 1045 |

## Last invocation (meta file)

| Item | Value |
|---|---|
| Status | finished |
| Started utc | 2026-11-02T07:04:07+00:00 |
| Finished utc | 2026-11-02T07:18:32+00:00 |
| Git commit | a4c33c40ac247f5fabcd5eb42375663ab0f2e5d3 |
| Git dirty | False |
| Generation frozen | True |
| Manifest sha256 matches | True |
| Prompt version | v3 |
| Prompt matches | True |
| Decoding | temperature: 0.0; top_p: 1.0; max_output_tokens: 2048; seed: 42; thinking_level: low |
| Requests | concurrency: 8; format_attempts: 3; timeout_seconds: 120; max_retries: 5; backoff_seconds: 2.0 |
| Adapter | adapter: GeminiAdapter; model_id: gemini-3.8-flash; library: google-genai; library_version: 2.34.0 |
