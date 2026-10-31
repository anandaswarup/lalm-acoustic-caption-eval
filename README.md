# LALM Acoustic Caption Evaluation

Evaluating Large Audio Language Model (LALM)-generated acoustic captions against human references on Clotho v2.1.

> **Private repository.** It remains private through double-blind review and is released only after acceptance (see `docs/experiment_plan.md`, E26).

## Research questions

- **RQ1:** How closely do model-generated captions agree with the five human reference captions?
- **RQ2:** How do model-generated captions compare with held-out human captions in reference-based metric scores and caption length?
- **RQ3:** What recurring patterns of agreement, additional detail, omission, and mismatch emerge when model-generated captions are compared with human references?

## Study design

| Item | Setting |
|---|---|
| Dataset | Clotho v2.1, complete evaluation split: 1,045 recordings × 5 human captions |
| Pilot | 20 recordings from the validation split (seed 42), technical validation only |
| Models | GPT-Audio-1.5, Gemini 3.8 Flash, Qwen3-Omni-30B-A3B-Instruct, Audio Flamingo 3 |
| Prompt | One fixed prompt for all models (`configs/generation.yaml`) |
| Metrics | BLEU-1–4, METEOR, CIDEr, SPICE, SPIDEr, SPIDEr-FL, BERTScore, FENSE; CLAP and MACE reported separately |

Full details: [`docs/proposal.md`](docs/proposal.md) and [`docs/experiment_plan.md`](docs/experiment_plan.md).

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
- **LaTeX** (`latexmk`, `pdflatex`) for `make paper`.
- **Hugging Face CLI** (`hf`) and `HF_TOKEN` for HF Jobs.

### Dataset

Clotho v2.1 ([Zenodo 4783391](https://zenodo.org/records/4783391)) must be available locally, with `CLOTHO_ROOT` in `.env` pointing to it. The expected layout is `audio/{split}/*.wav`, `captions/clotho_captions_{split}.csv`, `metadata/` and `manifests/`. Audio and raw dataset files are never committed.

### Compute placement

| Workload | Where |
|---|---|
| API inference (GPT-Audio-1.5, Gemini 3.8 Flash), lexical metrics, analysis, figures | Local laptop |
| Open-weight inference (Qwen3-Omni, Audio Flamingo 3), heavy model-based metrics | Hugging Face Jobs |

The `compute` field for each model is set in `configs/models.yaml`.

## Running experiments

Run `make help` for the full list. Main targets:

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
| `make paper` | Build the anonymized manuscript |
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
manuscript/   ICASSP LaTeX source (spconf), double-blind by default
docs/         proposal and experiment plan
```

## Manuscript

`manuscript/main.tex` uses the ICASSP `spconf.sty` / `IEEEbib.bst` template. `\blindtrue` (the default) produces the anonymized submission. Switch to `\blindfalse` only for the camera-ready version after acceptance.

## Status

| Phase | Experiments | Status |
|---|---|---|
| 1. Repository and dataset | E01–E03 | E01 done |
| 2. Caption generation | E04–E10 | — |
| 3. Automatic evaluation | E11–E12 | — |
| 4. Human calibration | E13–E16 | — |
| 5. Qualitative analysis | E17–E20 | — |
| 6. Synthesis and reproducibility | E21–E23 | — |
| 7. Manuscript and release | E24–E26 | — |
