# Run every target from the repository root. Python comes from the uv-managed
# .venv (built on the Miniforge interpreter); override with `make PYTHON=...`.

BASE_PYTHON ?= /opt/homebrew/Caskroom/miniforge/base/bin/python3
VENV        ?= .venv
PYTHON      ?= $(VENV)/bin/python
MODELS      := gpt_audio_1_5 gemini_3_8_flash qwen2_5_omni audio_flamingo_3

.DEFAULT_GOAL := help
.PHONY: help venv install check-env test lint format \
        prepare-data pilot generate hf-upload-audio fetch-hf consolidate \
        evaluate loo evaluate-four-ref lengths stats figures all paper paper-clean clean

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

lint: ## Lint and check formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

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
	$(PYTHON) scripts/evaluate_captions.py --set condition=five_reference

loo: ## Human leave-one-out calibration (E13)
	$(PYTHON) scripts/human_leave_one_out.py

evaluate-four-ref: ## Matched four-reference model evaluation (E14)
	$(PYTHON) scripts/evaluate_captions.py --set condition=four_reference

lengths: ## Caption-length analysis (E15)
	$(PYTHON) scripts/analyze_lengths.py

stats: ## Bootstrap CIs and cross-metric comparisons (E16, E21)
	$(PYTHON) scripts/analyze_results.py

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
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
