SHELL := sh
.ONESHELL:
.SHELLFLAGS := -eu -c
.DELETE_ON_ERROR:

# load environment config from .env if able
-include .env

ifndef PYTHON_ALIAS
	PYTHON_ALIAS := python
endif

SOURCES_PATH := src
TESTS_PATH := tests

.PHONY: install venv format lint test

# Setup virtual environment for local development.
venv:
	@echo '# Preparing development environment...'
	@echo '...preparing git hooks...'
	@cp -n ./config/pre-push ./.git/hooks/pre-push || :
	@echo '...preparing venv...'
	@$(PYTHON_ALIAS) -m venv .venv --prompt="VENV[DEV]" --clear --upgrade-deps
	@. ./.venv/bin/activate && pip install --upgrade pip && pip install --editable .[dev] --require-virtualenv 
	@echo '...development environment ready! Activate venv using `. ./.venv/bin/activate`.'

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

build: format lint test
