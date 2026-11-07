# Caption completion report: GPT-Audio-1.5 (evaluation)

Output: `outputs/captions/evaluation/gpt_audio_1_5_captions.jsonl` (SHA-256 `20be47d5bfebbf8b9d29f5983e55459cac335d8029b284d61526a71677b842d7`)

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
| Format retries (samples) | 148 |
| Fixed by a retry | 42 |
| Rejected responses | 268 |
| Still JSON-like | 106: `clotho_eval_0015`, `clotho_eval_0019`, `clotho_eval_0030`, `clotho_eval_0036`, `clotho_eval_0041`, `clotho_eval_0070`, `clotho_eval_0072`, `clotho_eval_0074`, `clotho_eval_0075`, `clotho_eval_0093`, `clotho_eval_0102`, `clotho_eval_0106`, `clotho_eval_0112`, `clotho_eval_0113`, `clotho_eval_0122`, `clotho_eval_0124`, `clotho_eval_0142`, `clotho_eval_0143`, `clotho_eval_0157`, `clotho_eval_0159` … (+86) |
| Of which exactly {"…"} (E09 unwraps) | 85: `clotho_eval_0015`, `clotho_eval_0036`, `clotho_eval_0041`, `clotho_eval_0070`, `clotho_eval_0075`, `clotho_eval_0093`, `clotho_eval_0112`, `clotho_eval_0113`, `clotho_eval_0122`, `clotho_eval_0124`, `clotho_eval_0142`, `clotho_eval_0143`, `clotho_eval_0159`, `clotho_eval_0162`, `clotho_eval_0168`, `clotho_eval_0182`, `clotho_eval_0221`, `clotho_eval_0222`, `clotho_eval_0234`, `clotho_eval_0248` … (+65) |
| Multi-line captions | none |
| Finish reasons | stop: 1045 |
| Caption words | min: 9; mean: 20.98; median: 21; max: 51 |

## Usage

| Item | Value |
|---|---|
| Tokens (ok responses) | input: 309950; input_audio: 233665; input_text: 76285; output: 27291; thinking: 0 |
| Peak GPU memory (GB) | n/a |
| Latency (s) | min: 1.596; mean: 4.35; median: 3.697; max: 56.58; total: 4544.779 |
| Samples with transient retries | 1 |
| Extra attempts | 1 |

## Model

| Item | Value |
|---|---|
| Model versions | gpt-audio-1.5: 1045 |
| System fingerprints | fp_142e5b87cb: 720; fp_80a31e8b83: 23; fp_3281f4f8fe: 94; fp_d86a785d62: 101; fp_bd509b8ec4: 58; fp_9d55025f37: 49 |
| Deviations | none |

## Last invocation (meta file)

| Item | Value |
|---|---|
| Status | finished |
| Started utc | 2026-11-02T06:35:42+00:00 |
| Finished utc | 2026-11-02T06:48:44+00:00 |
| Git commit | 588dd044a236e29f02ee2bc5c6ca55de96ad7173 |
| Git dirty | False |
| Generation frozen | True |
| Manifest sha256 matches | True |
| Prompt version | v3 |
| Prompt matches | True |
| Decoding | temperature: 0.0; top_p: 1.0; max_output_tokens: 2048; seed: 42 |
| Requests | concurrency: 8; format_attempts: 3; timeout_seconds: 120; max_retries: 5; backoff_seconds: 2.0 |
| Adapter | adapter: OpenAIAudioAdapter; model_id: gpt-audio-1.5; library: openai; library_version: 3.42.0 |
