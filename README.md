# LALM Acoustic Caption Evaluation

Evaluating Large Audio Language Model (LALM)-generated acoustic captions against human references on Clotho v2.1.

> **Private repository.** It remains private until the paper is submitted. It is then made public and linked from the paper, so that reviewers can inspect the code, experiments and artifacts.

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

#### Five-reference evaluation (E12)

`make evaluate` scores every model caption against all five references, with audio, on all 1,045 evaluation recordings (about 25 minutes on a laptop CPU). Malformed captions are scored as returned. It writes to `outputs/scores/`:

| File | Contents |
|---|---|
| `five_reference_scores.csv` | Per-recording score for every model and metric |
| `five_reference_corpus.csv` | Corpus value and item mean per model and metric |
| `five_reference_summary.csv` | Corpus value, item mean and 95% paired-bootstrap CI (1,000 resamples, seed 42), on all recordings and on the 1,024 where every model's caption is valid |
| `five_reference_differences.csv` | Paired item-mean differences for every model pair, with bootstrap CIs (uncorrected) |
| `five_reference_sensitivity.csv` | Whether the model order or a pairwise CI changes on the 1,024 valid recordings |

The report is `outputs/reports/five_reference_evaluation.{json,md}`. CLAP audio embeddings are cached in `outputs/cache/clap_audio.npz` (git-ignored) and reused by later tasks.

#### Human leave-one-out calibration (E13)

`make loo` holds out the human caption at each CSV position k = 1–5 and scores it against the other four references (subset `drop_k`, the same reference sets the models are scored against in E14), with audio, on all 1,045 recordings (about 16 minutes with the audio cache). Before scoring it checks, by position, that no held-out caption is in its own reference set and that the five held-out captions of each recording are exactly its references. The baseline averages the five positions: per recording, the mean of its position scores; the corpus value, the mean of the five per-position corpus values. Bootstrap resamples are the same as in E12 and E14.

Clotho's five captions of a recording often include near-paraphrase pairs, so a held-out caption can have a near-copy among its references, which a model caption never has. For each held-out caption the script records the word-set Jaccard (PTB tokens) with its closest other reference, and also reports the baseline on the held-out captions whose closest reference is below 0.6 and below 0.8 (`leave_one_out.overlap_thresholds` in `configs/metrics.yaml`), from the per-item scores without rescoring.

| File | Contents |
|---|---|
| `human_loo_scores.csv` | Per-recording score for every position and metric |
| `human_loo_corpus.csv` | Corpus value and item mean per position, and for the five-position mean |
| `human_loo_summary.csv` | Corpus value, item mean and 95% CI per position, for the five-position mean, and for the near-paraphrase sensitivity samples |
| `human_loo_overlap.csv` | Closest-reference word Jaccard of every held-out caption |

The report is `outputs/reports/human_loo_evaluation.{json,md}`.

#### Matched four-reference evaluation (E14)

`make evaluate-four-ref` scores every model caption against each subset `drop_k`, the same four references as the held-out human captions of E13, with audio, on all 1,045 recordings (20 scoring calls, about 72 minutes with the audio cache), and averages the five subsets as E13 does. The human baseline is read from E13's tables, not rescored; the script checks that they reproduce E13's report exactly. Model and human scores share references, metrics, aggregation and bootstrap resamples, so their differences are paired. Two sensitivity checks recompute the comparison without rescoring: on the 1,024 recordings where every model's caption is valid, and with E13's near-paraphrase masks (0.6 and 0.8) applied to model and human scores alike.

| File | Contents |
|---|---|
| `four_reference_scores.csv.gz` | Per-recording score for every model, subset and metric (gzip) |
| `four_reference_corpus.csv` | Corpus value and item mean per model and subset, and for the five-subset mean |
| `four_reference_summary.csv` | Corpus value, item mean and 95% CI per model and subset, and for the five-subset mean of models and human on every sensitivity sample |
| `four_reference_differences.csv` | Paired differences for every model pair on the five-subset mean, per sample |
| `matched_human_model_comparison.csv` | Paired model-minus-human differences on the five-subset mean, per sample |
| `four_reference_sensitivity.csv` | Whether the order (human included) or a pairwise CI changes on each sensitivity sample |

The report, including the model-minus-human calibration table, is `outputs/reports/four_reference_evaluation.{json,md}`.

#### Caption-length analysis (E15)

`make lengths` counts the words (whitespace tokens) of every scored model caption and human reference and reports, per system, the mean, SD, quartiles and range. For each model it gives the ratio of its mean length to the human mean and the per-recording difference from the mean length of the recording's five references, with bootstrap CIs over recordings (the same resamples as E12–E14), and the share of captions longer than the reference mean and than every reference. It also reports Spearman correlations between caption length and per-item score, using the five-reference model scores (E12) and the held-out human scores (E13, all five positions pooled, resampled by recording), for the metrics in `lengths.association_metrics` (`configs/metrics.yaml`). Everything is repeated on the 1,024 recordings where every model's caption is valid. It reads frozen files only and runs in seconds in `.venv`. Figures are drawn in E21 from the per-caption table.

| File | Contents |
|---|---|
| `caption_lengths_per_caption.csv` | Word count of every model and human caption, with the mean and maximum length of the references it is compared with |
| `caption_lengths_summary.csv` | Length statistics per system, ratio to human and difference from the reference mean with 95% CIs, per sample |
| `caption_lengths_associations.csv` | Spearman correlation between length (or length minus the reference mean) and score, with 95% CIs (uncorrected), per system, metric and sample |

The report is `outputs/reports/caption_length_analysis.{json,md}`.

#### Statistical comparison (E16)

`make stats` adds formal tests to the comparisons of E12–E15, from the committed per-item tables (no rescoring; about 7 minutes in `.venv`). Each comparison is paired over recordings and tested with a two-sided sign-flip permutation test of the mean difference (10,000 permutations), with a 95% bootstrap CI (10,000 resamples) and a paired effect size (d_z). Holm–Bonferroni correction is applied within four families fixed in `statistics` (`configs/metrics.yaml`), on all recordings: model pairs on the 14 main metrics (E12, 84 tests), models minus human on the matched four-reference scores (E14, 56 tests), model minus reference caption length (E15, 4 tests), and length–score Spearman correlations (E15, 35 tests, permutation over recordings). The E12–E14 sensitivity samples are reported uncorrected. Ranking stability is the share of bootstrap resamples in which each system takes each rank. The report documents the statistical assumptions and lists every decision that differs from the uncorrected marks of E12 and E14.

| File | Contents |
|---|---|
| `statistics_tests.csv` | Every paired test: difference, CI, d_z, p-value, Holm-adjusted p-value and decision, per family and sample |
| `statistics_correlations.csv` | Length–score Spearman correlations with CIs, permutation p-values and Holm decisions |
| `statistics_ranks.csv` | Share of bootstrap resamples at each rank, per system, metric and sample |
| `statistics_orders.csv` | Observed order per metric and the share of resamples that reproduce it |

The report is `outputs/reports/statistical_analysis.{json,md}`.

#### Qualitative review sample (E17)

`make qualitative-sample` draws 50 evaluation recordings at random (seed 42, `configs/qualitative.yaml`), independently of model scores, and prepares blinded review material for the 200 model captions. Each recording has its own random order of the four models behind the labels A–D, and the recordings are presented in random order under anonymous codes. The review sheets show the five references and each caption as scored, without file names (Clotho file names describe the content), model names or scores. The blinding key is not committed until the annotation labels are final (E18); its SHA-256 is recorded in the sample manifest so it can be checked when it is added.

| File (`outputs/qualitative/`) | Contents |
|---|---|
| `sample_manifest.json` | Seed, sampled recording IDs, reviewer templates, input checksums and the blinding key's SHA-256 |
| `review_items.csv` | One row per model caption: item ID, recording code, label, the five references and the caption |
| `review_sheet.md` | Reading view of the review items, one section per recording |
| `annotations_reviewer_1.csv` | Annotation template: agreement category, disputed caption phrases and notes (one reviewer; more are left for future work). `prepare_qualitative.py` never overwrites a template that holds labels |
| `categories.csv` | The six agreement categories |
| `blinding_key.csv` | Item to recording and model (git-ignored until the E18 labels are final) |

The report, with a descriptive comparison of the sample with all recordings, is `outputs/reports/qualitative_sample.{json,md}`.

#### Qualitative annotation (E18)

One reviewer labels each of the 200 captions with one of six agreement categories, following `outputs/qualitative/annotation_guidelines.md`: the caption is compared with the five references only, without audio, and principal content is what at least two references mention. Caption phrases that no reference supports are listed in `disputed_details`; after the labels are final, every caption is also rated against the audio in E19. Once the labels are final, `make annotations` checks the template (every item labelled, review columns unchanged), verifies the blinding key against the SHA-256 in the sample manifest, unblinds the labels and writes:

| File (`outputs/qualitative/`) | Contents |
|---|---|
| `qualitative_annotations.csv` | Unblinded labels: item, recording, model, category, disputed details and notes |
| `category_summary.csv` | Count and share of each category per model, with Wilson 95% intervals |
| `disputed_summary.csv` | Captions with disputed details and number of disputed phrases per model |
| `scores_by_category.csv` | Mean five-reference item score per category (descriptive) |

The report is `outputs/reports/qualitative_annotation.{json,md}`.

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
| `make qualitative-sample` | E17: blinded qualitative review sample |
| `make annotations` | E18: check, unblind and summarize the annotations |
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
| | E12 | Five-reference model evaluation | Done |
| 4. Human calibration | E13 | Human leave-one-out calibration | Done |
| | E14 | Matched four-reference model evaluation | Done |
| | E15 | Caption-length analysis | Done |
| | E16 | Statistical comparison and uncertainty | Done |
| 5. Qualitative analysis | E17 | Select qualitative review sample | Done |
| | E18 | Annotate model–reference agreement | — |
| | E19 | Audio-grounded review of the qualitative sample | — |
| | E20 | Representative caption comparisons | — |
| 6. Synthesis and reproducibility | E21 | Cross-metric and model comparison | — |
| | E22 | Reproduce results from frozen artifacts | — |
| | E23 | Finalize research repository | — |
| 7. Manuscript and release | E24 | Draft manuscript | — |
| | E25 | Final validation, public release and submission | — |
| | E26 | Camera-ready and versioned release after acceptance | — |
