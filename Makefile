# CaliPPer Makefile — convenience targets
# Usage: make <target>
#   make help            # this message
#   make install         # create conda env + pip install -e .
#   make install-pip     # pip-only fallback (no conda)
#   make test            # pytest tests/
#   make lint            # ruff + black --check
#   make format          # black + ruff --fix (writes changes)
#   make verify-env      # pre-flight: Python, packages, data, GPU
#   make reproduce-all   # regenerate all 5 figure values + verify
#   make reproduce-fig FIG=2   # regenerate just one figure (any of 2,3,4,5,6)
#   make verify          # verify regenerated values against committed reference
#   make clean           # remove pycache, build artefacts, test outputs

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---- conda env name (override with: make ENV=other-env install) ----
ENV ?= calipper

# ---- targets ----

.PHONY: help
help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## create conda env from environment.yml + pip install -e .
	conda env create -f environment.yml -n $(ENV) || conda env update -f environment.yml -n $(ENV)
	conda run -n $(ENV) pip install -e .
	@echo
	@echo "Done. Activate with: conda activate $(ENV)"

.PHONY: install-pip
install-pip:  ## pip-only install (no conda)
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .

.PHONY: test
test:  ## run pytest test suite
	pytest tests/ -v

.PHONY: lint
lint:  ## check formatting + lint
	ruff check calipper tests reproduce/scripts
	black --check calipper tests reproduce/scripts

.PHONY: format
format:  ## apply black + ruff --fix
	black calipper tests reproduce/scripts
	ruff check --fix calipper tests reproduce/scripts

.PHONY: verify-env
verify-env:  ## pre-flight check (Python, packages, calipper, data, GPU)
	bash reproduce/verify_environment.sh

.PHONY: reproduce-all
reproduce-all:  ## regenerate all 5 figures + run verify.sh
	bash reproduce/reproduce.sh

.PHONY: reproduce-fig
reproduce-fig:  ## regenerate one figure: make reproduce-fig FIG=2
	@if [ -z "$(FIG)" ]; then echo "Usage: make reproduce-fig FIG=2"; exit 1; fi
	bash reproduce/reproduce.sh --figure $(FIG)

.PHONY: verify
verify:  ## verify regenerated values against reference (numerical equality 1e-10)
	bash reproduce/verify.sh

.PHONY: clean
clean:  ## remove pycache, build artefacts, test outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/ .coverage
	rm -rf reproduce/data/output/ reproduce/figures/output/ reproduce/verification.json
	@echo "Cleaned. (data/input/ and downloaded weights are preserved; use 'make clean-data' to wipe those too)"

.PHONY: clean-data
clean-data:  ## also wipe downloaded data + weights (forces re-download)
	rm -rf reproduce/data/input/
	rm -rf models/*/weights/ models/retrospective/*/weights/
