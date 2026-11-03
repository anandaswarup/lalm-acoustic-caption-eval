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
| Models | GPT-Audio-1.5, Gemini 3.8 Flash, Qwen2.5-Omni-7B, Audio Flamingo 3 |
| Prompt | One fixed prompt for all models (`configs/generation.yaml`) |
| Metrics | BLEU-1–4, METEOR, CIDEr-D, SPICE, SPIDEr, SPIDEr-FL, FENSE, BERTScore-F1 (raw and rescaled); CLAP similarity and MACE reported separately |

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

- A separate metrics environment and pinned tools (JDK 11, jars, checkpoints) for the caption metrics; see [Metrics](#metrics).
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
| API inference (GPT-Audio-1.5, Gemini 3.8 Flash), all caption metrics (CPU), analysis, figures | Local laptop |
| Open-weight inference (Qwen2.5-Omni-7B, Audio Flamingo 3) | Hugging Face Jobs |

The `compute` field for each model is set in `configs/models.yaml`. The E11 validation estimates about 2 hours of CPU time for all metrics on all evaluation conditions (E12–E14), so the metrics run on the laptop.

### Metrics

The caption metrics run in their own environment, `.venv-metrics` (`requirements-metrics.txt`), because MS-CLAP (used for CLAP similarity and MACE) needs older numpy, pandas and transformers than the rest of the project. METEOR, SPICE and the PTB tokenizer need **Java 11–13**: aac-metrics rejects newer versions and SPICE needs the Nashorn engine, removed in Java 15. The metrics therefore use their own JDK 11, downloaded into `tools/`, not the system Java.

```sh
make venv-metrics install-metrics   # .venv-metrics from requirements-metrics.txt
make metric-tools       # JDK 11, metric jars and model checkpoints into tools/ (about 4.5 GB, git-ignored)
make test-metrics       # integration tests with the real jars and checkpoints
make validate-metrics   # pilot validation; writes outputs/reports/metric_implementation.{json,md}
```

- `make metric-tools` checks every download against the SHA-256 or commit pinned in `configs/metrics.yaml` (`tools:`) and records the SHA-256 of every file in `tools/MANIFEST.json`; `python scripts/setup_metric_tools.py --check` verifies them again. CoreNLP 3.6.0 (for SPICE) comes from Stanford's mirror `downloads.cs.stanford.edu`, because the URL aac-metrics uses is unreachable.
- Scoring runs offline: models load only from `tools/` at the pinned revisions.
- Every text metric scores PTB-tokenized text (lowercased, punctuation removed), as in aac-metrics and DCASE. Each scoring call covers a whole condition at once, because CIDEr-D and the BLEU and METEOR corpus statistics depend on every reference in the call.
- Metric parameters are in `configs/metrics.yaml` (`settings:`). Implementation notes:
  - BLEU, METEOR, CIDEr-D, SBERT similarity, the fluency-error detector and BERTScore come from aac-metrics 0.5.5.
  - SPICE runs without its parse cache, whose native library is x86-only; scores are unchanged.
  - BERTScore uses roberta-large's last layer (24), as aac-metrics does, with `max_length` 128. The rescaled F1 uses the layer-24 baseline.
  - CLAP similarity and MACE use MS-CLAP 2023 on the whole clip (7 s chunks, duration-weighted), and MACE combines audio–text and text–text similarity, as in the MACE paper.
- The implementation report covers versions, pins, determinism, truncation, timings, and the interpretation and limitations of each metric.

## Running tasks

Run `make help` for the full list. Main targets (task IDs are listed under [Tasks](#tasks)):

| Target | Task |
|---|---|
| `make check-env` | Environment validation |
| `make prepare-data` | E02–E03: validate Clotho, build frozen manifests |
| `make pilot` | E05: pilot generation |
| `make generate` / `make generate-<model>` | E06–E09: caption generation |
| `make consolidate` | E10: merge and freeze captions |
| `make metric-tools` / `make test-metrics` / `make validate-metrics` | E11: metric tools, integration tests, validation report |
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

## Tasks

Tasks are numbered E01–E26. Configs, scripts, Makefile targets and commit messages refer to these numbers.

| Phase | ID | Task | Status |
|---|---|---|---|
| 1. Repository and dataset | E01 | Scaffold repository and execution environment | Done |
| | E02 | Download, validate and index Clotho v2.1 | Done |
| | E03 | Create and freeze the evaluation manifest | Done |
| 2. Caption generation | E04 | Unified LALM inference interfaces | Done |
| | E05 | Pilot generation and configuration freeze | Done |
| | E06 | Captions: GPT-Audio-1.5 | Done |
| | E07 | Captions: Gemini 3.8 Flash | Done |
| | E08 | Captions: Qwen2.5-Omni-7B | Done |
| | E09 | Captions: Audio Flamingo 3 | Done |
| | E10 | Consolidate and freeze model captions | Done |
| 3. Automatic evaluation | E11 | Implement and validate caption metrics | Done |
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
