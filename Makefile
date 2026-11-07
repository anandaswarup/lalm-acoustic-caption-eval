# Run every target from the repository root. Python comes from the uv-managed
# .venv (built on the Miniforge interpreter); override with `make PYTHON=...`.
# The caption metrics (E11-E14) run in a separate .venv-metrics (see requirements-metrics.txt).

BASE_PYTHON ?= /opt/homebrew/Caskroom/miniforge/base/bin/python3
VENV        ?= .venv
PYTHON      ?= $(VENV)/bin/python
METRICS_VENV   ?= .venv-metrics
METRICS_PYTHON ?= $(METRICS_VENV)/bin/python
MODELS      := gpt_audio_1_5 gemini_3_8_flash qwen2_5_omni audio_flamingo_3

.DEFAULT_GOAL := help
.PHONY: help venv install check-env test test-metrics validate-metrics lint format venv-metrics install-metrics metric-tools \
        prepare-data pilot generate hf-upload-audio fetch-hf consolidate \
        evaluate loo evaluate-four-ref lengths stats qualitative-sample annotations audio-review-sheet audio-review examples compare figures all paper paper-clean clean

help: ## List available targets
	@grep -E '^[a-zA-Z0-9_%-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# --- Environment (E01, E23) ---------------------------------------------------
venv: ## Create .venv with uv from the Miniforge Python
	uv venv --python $(BASE_PYTHON) $(VENV)

install: ## Install pinned dependencies into .venv
	uv pip install --python $(PYTHON) -r requirements.txt

check-env: ## Validate Python, dependencies, configs, tools, dataset and credentials
	$(PYTHON) scripts/check_environment.py

test: ## Run the test suite
	$(PYTHON) -m pytest

test-metrics: ## Metric integration tests with the real jars and checkpoints (E11; needs make metric-tools)
	$(METRICS_PYTHON) -m pytest -m metrics tests/test_metrics_integration.py

validate-metrics: ## Validate the metrics on the pilot and write the metric implementation report (E11)
	$(METRICS_PYTHON) scripts/validate_metrics.py

lint: ## Lint and check formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

venv-metrics: ## Create .venv-metrics for the caption metrics (E11)
	uv venv --python $(BASE_PYTHON) $(METRICS_VENV)

install-metrics: ## Install pinned metric dependencies into .venv-metrics
	uv pip install --python $(METRICS_PYTHON) -r requirements-metrics.txt

metric-tools: ## Download and verify the JDK 11, metric jars and checkpoints into tools/ (E11)
	$(METRICS_PYTHON) scripts/setup_metric_tools.py

# --- Phase 1: dataset (E02-E03) ----------------------------------------------
prepare-data: ## Validate Clotho and build frozen evaluation/pilot manifests
	$(PYTHON) scripts/prepare_dataset.py

# --- Phase 2: caption generation (E04-E10) -----------------------------------
# Models with compute: hf_jobs are submitted to Hugging Face Jobs (fetch results with `make fetch-hf`);
# the others run locally.
define run_model
if [ "$$($(PYTHON) scripts/hf_jobs.py compute $(1))" = hf_jobs ]; then \
	$(PYTHON) scripts/hf_jobs.py submit --model $(1) --run $(2); \
else \
	$(PYTHON) scripts/generate_captions.py --model $(1) --run $(2); \
fi
endef

pilot: ## Pilot generation on 20 validation-split recordings (E05)
	$(foreach m,$(MODELS),$(call run_model,$(m),pilot) &&) true

generate-%: ## Generate captions for one model, e.g. make generate-qwen2_5_omni
	$(call run_model,$*,evaluation)

hf-upload-audio: ## Upload the manifest WAVs to the private HF bucket (once)
	$(PYTHON) scripts/hf_jobs.py upload-audio

fetch-hf: ## Pull Hugging Face Jobs caption outputs into outputs/captions/
	$(PYTHON) scripts/hf_jobs.py fetch

generate: $(addprefix generate-,$(MODELS)) ## Generate captions for all four models

consolidate: ## Merge and freeze model captions (E10)
	$(PYTHON) scripts/consolidate_captions.py

# --- Phases 3-4: evaluation (E11-E16) ----------------------------------------
evaluate: ## Five-reference evaluation (E12)
	$(METRICS_PYTHON) scripts/evaluate_captions.py --set condition=five_reference

loo: ## Human leave-one-out calibration (E13)
	$(METRICS_PYTHON) scripts/human_leave_one_out.py

evaluate-four-ref: ## Matched four-reference model evaluation (E14)
	$(METRICS_PYTHON) scripts/evaluate_captions.py --set condition=four_reference

lengths: ## Caption-length analysis (E15)
	$(PYTHON) scripts/analyze_lengths.py

stats: ## Bootstrap CIs and cross-metric comparisons (E16, E21)
	$(PYTHON) scripts/analyze_results.py

qualitative-sample: ## Blinded qualitative review sample and annotation templates (E17)
	$(PYTHON) scripts/prepare_qualitative.py

annotations: ## Check, unblind and summarize the qualitative annotations (E18)
	$(PYTHON) scripts/analyze_annotations.py

audio-review-sheet: ## Blinded audio-review material and templates for the qualitative sample (E19)
	$(PYTHON) scripts/prepare_audio_review.py

audio-review: ## Check, unblind and summarize the audio-grounded review (E19)
	$(PYTHON) scripts/analyze_audio_review.py

examples: ## Representative caption comparisons, tables and qualitative figures (E20)
	$(PYTHON) scripts/prepare_examples.py

compare: ## Cross-metric correlations, model rankings and audio-rating concordance (E21)
	$(PYTHON) scripts/compare_metrics.py

figures: ## Regenerate manuscript tables and figures
	$(PYTHON) scripts/generate_figures.py

all: prepare-data generate consolidate evaluate loo evaluate-four-ref lengths stats figures ## Full workflow

# --- Manuscript (E24-E25) ----------------------------------------------------
paper: ## Build the manuscript PDF (once manuscript/main.tex exists)
	@test -f manuscript/main.tex || { echo "manuscript/main.tex not found; add the template first."; exit 1; }
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-clean: ## Remove LaTeX build artifacts
	cd manuscript && latexmk -c main.tex

clean: ## Remove caches and logs (keeps outputs and manuscript PDF)
	rm -rf .pytest_cache .ruff_cache outputs/logs
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -not -path './$(METRICS_VENV)/*' -not -path './tools/*' -exec rm -rf {} +
