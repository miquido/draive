SHELL := sh
.ONESHELL:
.SHELLFLAGS := -eu -c
.DELETE_ON_ERROR:

SOURCES_PATH := src
TESTS_PATH := tests

# load environment config from .env if able
-include .env

ifndef PYTHON_ALIAS
	PYTHON_ALIAS := python
endif

.PHONY: venv sync lock update format lint test

# Setup virtual environment for local development.
venv:
	@echo '# Preparing development environment...'
	@echo '...cloning .env...'
	@cp -n ./config/.env.example ./.env || :
	@echo '...preparing git hooks...'
	@cp -n ./config/pre-push ./.git/hooks/pre-push || :
	@echo '...preparing venv...'
	@$(PYTHON_ALIAS) -m venv .venv --prompt="VENV[DEV]" --clear --upgrade-deps
	@. ./.venv/bin/activate && pip install --upgrade pip && pip install --editable .[dev] --require-virtualenv -c constraints.txt
	@echo '...development environment ready! Activate venv using `. ./.venv/bin/activate`.'

# Sync environment with uv based on constraints.txt
sync:
	@uv pip install --editable .[dev] --constraint constraints.txt

# Generate a set of locked dependencies from pyproject.toml
lock:
	@uv pip compile pyproject.toml -o constraints.txt --all-extras

# Update and lock dependencies from pyproject.toml
update:
	@uv pip compile pyproject.toml -o constraints.txt --all-extras --upgrade
	@uv pip install --editable .[dev] --constraint constraints.txt

# Run formatter.
format:
	@ruff --quiet --fix $(SOURCES_PATH) $(TESTS_PATH)

# Run linters and code checks.
lint:
	@bandit -r $(SOURCES_PATH)
	@ruff $(SOURCES_PATH) $(TESTS_PATH)
	@pyright --project ./

# Run tests suite.
test:
	@$(PYTHON_ALIAS) -B -m pytest -v --cov=$(SOURCES_PATH) --rootdir=$(TESTS_PATH)
