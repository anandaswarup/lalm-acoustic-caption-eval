# Run every target from the repository root. Python comes from the uv-managed
# .venv (built on the Miniforge interpreter); override with `make PYTHON=...`.
# The caption metrics (E10-E13) run in a separate .venv-metrics (see requirements-metrics.txt).

BASE_PYTHON ?= /opt/homebrew/Caskroom/miniforge/base/bin/python3
VENV        ?= .venv
PYTHON      ?= $(VENV)/bin/python
METRICS_VENV   ?= .venv-metrics
METRICS_PYTHON ?= $(METRICS_VENV)/bin/python
MODELS      := gpt_audio_1_5 gemini_3_8_flash qwen2_5_omni audio_flamingo_3

.DEFAULT_GOAL := help
.PHONY: help venv install check-env test test-metrics lint format venv-metrics install-metrics metric-tools \
        download-data prepare-data generate hf-upload-audio fetch-hf consolidate \
        evaluate loo evaluate-four-ref lengths stats qualitative-sample annotations audio-review-sheet audio-review examples compare figures data captions scores qualitative all paper paper-clean clean

help: ## List available targets
	@grep -E '^[a-zA-Z0-9_%-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# --- Environment (E01) --------------------------------------------------------
venv: ## Create .venv with uv from the Miniforge Python
	uv venv --python $(BASE_PYTHON) $(VENV)

install: ## Install pinned dependencies into .venv
	uv pip install --python $(PYTHON) -r requirements.txt

check-env: ## Validate Python, dependencies, configs, tools, dataset and credentials
	$(PYTHON) scripts/check_environment.py

test: ## Run the test suite
	$(PYTHON) -m pytest

test-metrics: ## Metric integration tests with the real jars and checkpoints (E10; needs make metric-tools)
	$(METRICS_PYTHON) -m pytest -m metrics tests/test_metrics_integration.py

lint: ## Lint and check formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

venv-metrics: ## Create .venv-metrics for the caption metrics (E10)
	uv venv --python $(BASE_PYTHON) $(METRICS_VENV)

install-metrics: ## Install pinned metric dependencies into .venv-metrics
	uv pip install --python $(METRICS_PYTHON) -r requirements-metrics.txt

metric-tools: ## Download and verify the JDK 11, metric jars and checkpoints into tools/ (E10)
	$(METRICS_PYTHON) scripts/setup_metric_tools.py

# --- Phase 1: dataset (E02-E03) ----------------------------------------------
download-data: ## Download, verify, extract and index Clotho v2.1 into CLOTHO_ROOT (evaluation audio)
	$(PYTHON) scripts/download_clotho.py

prepare-data: ## Validate Clotho and build the frozen evaluation manifests
	$(PYTHON) scripts/prepare_dataset.py

# --- Phase 2: caption generation (E04-E09) -----------------------------------
# Models with compute: hf_jobs are submitted to Hugging Face Jobs (fetch results with `make fetch-hf`);
# the others run locally.
define run_model
if [ "$$($(PYTHON) scripts/hf_jobs.py compute $(1))" = hf_jobs ]; then \
	$(PYTHON) scripts/hf_jobs.py submit --model $(1) --run $(2); \
else \
	$(PYTHON) scripts/generate_captions.py --model $(1) --run $(2); \
fi
endef

generate-%: ## Generate captions for one model, e.g. make generate-qwen2_5_omni
	$(call run_model,$*,evaluation)

hf-upload-audio: ## Upload the manifest WAVs to the private HF bucket (once)
	$(PYTHON) scripts/hf_jobs.py upload-audio

fetch-hf: ## Pull Hugging Face Jobs caption outputs into outputs/captions/
	$(PYTHON) scripts/hf_jobs.py fetch

generate: $(addprefix generate-,$(MODELS)) ## Generate captions for all four models

consolidate: ## Merge and freeze model captions (E09)
	$(PYTHON) scripts/consolidate_captions.py

# --- Phases 3-4: evaluation (E10-E15) ----------------------------------------
evaluate: ## Five-reference evaluation (E11)
	$(METRICS_PYTHON) scripts/evaluate_captions.py --set condition=five_reference

loo: ## Human leave-one-out calibration (E12)
	$(METRICS_PYTHON) scripts/human_leave_one_out.py

evaluate-four-ref: ## Matched four-reference model evaluation (E13)
	$(METRICS_PYTHON) scripts/evaluate_captions.py --set condition=four_reference

lengths: ## Caption-length analysis (E14)
	$(PYTHON) scripts/analyze_lengths.py

stats: ## Formal tests, bootstrap CIs and ranking stability (E15)
	$(PYTHON) scripts/analyze_results.py

qualitative-sample: ## Blinded qualitative review sample and annotation templates (E16)
	$(PYTHON) scripts/prepare_qualitative.py

annotations: ## Check, unblind and summarize the qualitative annotations (E17)
	$(PYTHON) scripts/analyze_annotations.py

audio-review-sheet: ## Blinded audio-review material and templates for the qualitative sample (E18)
	$(PYTHON) scripts/prepare_audio_review.py

audio-review: ## Check, unblind and summarize the audio-grounded review (E18)
	$(PYTHON) scripts/analyze_audio_review.py

examples: ## Representative caption comparisons, tables and qualitative figures (E19)
	$(PYTHON) scripts/prepare_examples.py

compare: ## Cross-metric correlations, model rankings and audio-rating concordance (E20)
	$(PYTHON) scripts/compare_metrics.py

figures: ## E11 scores and E13 model - human figures (E20)
	$(PYTHON) scripts/generate_figures.py

# --- Complete workflow -------------------------------------------------------
# Four stages with two pauses: Hugging Face Jobs run detached (wait for them, then `make fetch-hf`), and the
# qualitative review needs human labels (E17, E18). Each stage runs its steps in the order listed.
data: download-data prepare-data ## Stage 1: Clotho download and frozen manifests (E02-E03)

captions: generate ## Stage 2: evaluation captions (E05-E08); HF Jobs are detached, then make fetch-hf

scores: consolidate evaluate loo evaluate-four-ref lengths stats figures ## Stage 3: E09-E15, E20 figures

qualitative: annotations audio-review examples compare ## Stage 4: E17-E20 analyses, once the review labels exist

all: data captions ## Stages 1-2, then lists the remaining steps
	@echo ""
	@echo "Stages 1-2 done. The Qwen2.5-Omni and Audio Flamingo 3 captions are being generated on Hugging Face Jobs."
	@echo "When the jobs have finished:   make fetch-hf scores"
	@echo "Qualitative review:            make qualitative-sample, label outputs/qualitative/annotations_reviewer_1.csv (E17),"
	@echo "                               make audio-review-sheet, label the two audio-review templates (E18),"
	@echo "                               make qualitative"

# --- Manuscript --------------------------------------------------------------
paper: ## Build the manuscript PDF (once manuscript/main.tex exists)
	@test -f manuscript/main.tex || { echo "manuscript/main.tex not found; add the template first."; exit 1; }
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-clean: ## Remove LaTeX build artifacts
	cd manuscript && latexmk -c main.tex

clean: ## Remove caches and logs (keeps outputs and manuscript PDF)
	rm -rf .pytest_cache .ruff_cache outputs/logs
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -not -path './$(METRICS_VENV)/*' -not -path './tools/*' -exec rm -rf {} +
