# Caption validation report (E09)

## Dataset

| Item | Value |
|---|---|
| Recordings | 1045 |
| Models | gpt_audio_1_5, gemini_3_8_flash, qwen2_5_omni, audio_flamingo_3 |
| Model captions | 4180 |
| Human references | 5225 |
| Recordings valid for all models (sensitivity check) | 1024 |
| Exclusions | none (malformed captions are scored as returned) |
| Prompt SHA-256 | `6a50f6d3766d4d1bdfcbc16382d99f69e91158c2d1156cb9da03973293351cbb` |

## Normalization per model

| Model | none | unwrapped | malformed | Malformed types | Words min / mean / max | Distinct |
|---|---|---|---|---|---|---|
| gpt_audio_1_5 | 939 | 85 | 21 | fields: 17, refusal: 1, result: 3 | 9 / 20.98 / 51 | 1030 |
| gemini_3_8_flash | 1045 | 0 | 0 | none | 8 / 16.36 / 36 | 1044 |
| qwen2_5_omni | 1045 | 0 | 0 | none | 4 / 16.96 / 33 | 1018 |
| audio_flamingo_3 | 1045 | 0 | 0 | none | 4 / 18.96 / 44 | 1023 |

## Malformed captions: gpt_audio_1_5

| sample_id | Type |
|---|---|
| `clotho_eval_0019` | fields |
| `clotho_eval_0030` | result |
| `clotho_eval_0072` | fields |
| `clotho_eval_0074` | fields |
| `clotho_eval_0102` | fields |
| `clotho_eval_0106` | fields |
| `clotho_eval_0157` | fields |
| `clotho_eval_0181` | fields |
| `clotho_eval_0207` | fields |
| `clotho_eval_0323` | refusal |
| `clotho_eval_0482` | result |
| `clotho_eval_0490` | fields |
| `clotho_eval_0584` | fields |
| `clotho_eval_0599` | fields |
| `clotho_eval_0626` | fields |
| `clotho_eval_0634` | fields |
| `clotho_eval_0675` | result |
| `clotho_eval_0699` | fields |
| `clotho_eval_0716` | fields |
| `clotho_eval_0736` | fields |
| `clotho_eval_0784` | fields |

## SHA-256

| File | SHA-256 |
|---|---|
| `all_model_captions.jsonl` | `10ef8b655f3941911c2834d41ab0a0622c8f894d4520132df95439dd0beb3b02` |
| `evaluation/audio_flamingo_3_captions.jsonl` | `eaa018aecf42b4f052ce94924004c9f44108dafd3178209bb441c61090796e0a` |
| `evaluation/gemini_3_8_flash_captions.jsonl` | `2c020d6b6121156f0b6c3d0c968f970a8f62220bd4d15eee123f310df50e2c93` |
| `evaluation/gpt_audio_1_5_captions.jsonl` | `20be47d5bfebbf8b9d29f5983e55459cac335d8029b284d61526a71677b842d7` |
| `evaluation/qwen2_5_omni_captions.jsonl` | `cd64fef2ce242a96774bd09bd5471d12d98adbbe5a3c1e047744daf0a5749552` |
