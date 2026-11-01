# LALM Acoustic Caption Evaluation

Evaluating Large Audio Language Model (LALM)-generated acoustic captions against human references on Clotho v2.1.

> **Private repository.** It remains private through double-blind review and is released only after acceptance.

## Research questions

- **RQ1:** How closely do model-generated captions agree with the five human reference captions?
- **RQ2:** How do model-generated captions compare with held-out human captions in reference-based metric scores and caption length?
- **RQ3:** What recurring patterns of agreement, additional detail, omission, and mismatch emerge when model-generated captions are compared with human references?

## Study design

| Item | Setting |
|---|---|
| Dataset | Clotho v2.1, complete evaluation split: 1,045 recordings × 5 human captions |
| Pilot | 20 recordings from the validation split (seed 42), excluding recordings that share a Freesound source with the evaluation split; technical validation only |
| Models | GPT-Audio-1.5, Gemini 3.8 Flash, Qwen3-Omni-30B-A3B-Instruct, Audio Flamingo 3 |
| Prompt | One fixed prompt for all models (`configs/generation.yaml`) |
| Metrics | BLEU-1–4, METEOR, CIDEr, SPICE, SPIDEr, SPIDEr-FL, BERTScore, FENSE; CLAP and MACE reported separately |

## Setup

The project is **not** installed as a package. Scripts run from the repository root. The Python interpreter comes from Miniforge, and the virtual environment and dependencies are managed with [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:anandaswarup/lalm-acoustic-caption-eval.git
cd lalm-acoustic-caption-eval

make venv          # uv venv from the Miniforge Python (override: make venv BASE_PYTHON=/path/to/python3)
make install       # uv pip install -r requirements.txt
cp .env.example .env   # then set CLOTHO_ROOT and credentials
make check-env
make test
```

Other requirements:

- **Java** for METEOR, SPICE and the PTB tokenizer (E11).
- **LaTeX** (`latexmk`, `pdflatex`) for `make paper`, once the manuscript is added.
- **Hugging Face CLI** (`hf`) and `HF_TOKEN` for HF Jobs.

### Dataset

Clotho v2.1 ([Zenodo 4783391](https://zenodo.org/records/4783391)) must be available locally, with `CLOTHO_ROOT` in `.env` pointing to it. The expected layout is `audio/{split}/*.wav`, `captions/clotho_captions_{split}.csv`, `metadata/` and `manifests/`. Audio and raw dataset files are never committed.

### Frozen manifests

`make prepare-data` validates the local Clotho copy and writes the frozen manifests to `data/manifests/`:

| File | Contents |
|---|---|
| `clotho_eval_manifest.jsonl` | 1,045 evaluation recordings sorted by file name (case-sensitive code-point order), with stable IDs `clotho_eval_0000`–`clotho_eval_1044`, audio properties, audio SHA-256 and Freesound licence |
| `clotho_eval_references.json` | `sample_id` → five human captions, in their original order (5,225 in total) |
| `clotho_pilot_manifest.jsonl` | 20 pilot recordings from the validation split, with captions |
| `CLOTHO_LICENSE` | Clotho caption copyright notice, which must accompany every copy of the captions |
| `SHA256SUMS` | Checksums of the files above |

Validation and manifest reports are in `outputs/reports/`. The script does not overwrite existing manifests unless given `--force`, and `python scripts/prepare_dataset.py --check` checks them against `SHA256SUMS`.

### Compute placement

| Workload | Where |
|---|---|
| API inference (GPT-Audio-1.5, Gemini 3.8 Flash), lexical metrics, analysis, figures | Local laptop |
| Open-weight inference (Qwen3-Omni, Audio Flamingo 3), heavy model-based metrics | Hugging Face Jobs |

The `compute` field for each model is set in `configs/models.yaml`.

## Running experiments

Run `make help` for the full list. Main targets (experiment IDs are listed under [Experiments](#experiments)):

| Target | Experiment |
|---|---|
| `make check-env` | Environment validation |
| `make prepare-data` | E02–E03: validate Clotho, build frozen manifests |
| `make pilot` | E05: pilot generation |
| `make generate` / `make generate-<model>` | E06–E09: caption generation |
| `make consolidate` | E10: merge and freeze captions |
| `make evaluate` | E12: five-reference evaluation |
| `make loo` | E13: human leave-one-out calibration |
| `make evaluate-four-ref` | E14: matched four-reference evaluation |
| `make lengths` | E15: caption-length analysis |
| `make stats` | E16, E21: bootstrap CIs and cross-metric comparison |
| `make figures` | Tables and figures |
| `make paper` | Build the manuscript (once added) |
| `make all` | Complete workflow |

Every script accepts `--seed`, `--log-level`, `--no-log-file` and repeatable `--set key=value` config overrides. Run logs, including the git commit and environment details, go to `outputs/logs/`.

## Repository layout

```text
configs/      dataset, model, generation and metric configurations
src/          library code (data, models, generation, metrics, evaluation, analysis, utils)
scripts/      standalone entry points, run from the repository root
tests/        pytest suite
data/         frozen manifests (no audio)
outputs/      captions, scores, qualitative annotations, tables, figures
manuscript/   LaTeX manuscript (added when writing the paper)
```

## Experiments

Experiments are numbered E01–E26. Configs, scripts, Makefile targets and commit messages refer to these numbers.

| Phase | ID | Experiment | Status |
|---|---|---|---|
| 1. Repository and dataset | E01 | Scaffold repository and execution environment | Done |
| | E02 | Download, validate and index Clotho v2.1 | Done |
| | E03 | Create and freeze the evaluation manifest | Done |
| 2. Caption generation | E04 | Unified LALM inference interfaces | — |
| | E05 | Pilot generation and configuration freeze | — |
| | E06 | Captions: GPT-Audio-1.5 | — |
| | E07 | Captions: Gemini 3.8 Flash | — |
| | E08 | Captions: Qwen3-Omni | — |
| | E09 | Captions: Audio Flamingo 3 | — |
| | E10 | Consolidate and freeze model captions | — |
| 3. Automatic evaluation | E11 | Implement and validate caption metrics | — |
| | E12 | Five-reference model evaluation | — |
| 4. Human calibration | E13 | Human leave-one-out calibration | — |
| | E14 | Matched four-reference model evaluation | — |
| | E15 | Caption-length analysis | — |
| | E16 | Statistical comparison and uncertainty | — |
| 5. Qualitative analysis | E17 | Select qualitative review sample | — |
| | E18 | Annotate model–reference agreement | — |
| | E19 | Audio-grounded verification of selected cases | — |
| | E20 | Representative caption comparisons | — |
| 6. Synthesis and reproducibility | E21 | Cross-metric and model comparison | — |
| | E22 | Reproduce results from frozen artifacts | — |
| | E23 | Finalize research repository | — |
| 7. Manuscript and release | E24 | Draft manuscript | — |
| | E25 | Final validation and double-blind submission | — |
| | E26 | Public release after acceptance | — |
