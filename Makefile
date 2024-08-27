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

ifndef INSTALL_OPTIONS
	INSTALL_OPTIONS := .[dev]
endif

ifndef UV_VERSION
	UV_VERSION := 0.3.3
endif

.PHONY: install venv sync lock update format lint test release

# Install in system without virtual environment and extras. DO NOT USE FOR DEVELOPMENT
install:
	@echo '# Installing...'
	@echo '...installing uv...'
	@curl -LsSf https://github.com/astral-sh/uv/releases/download/$(UV_VERSION)/uv-installer.sh | sh
	@echo '...preparing dependencies...'
	@uv pip install $(INSTALL_OPTIONS) --system --constraint constraints
	@echo '...finished!'

# Setup virtual environment for local development.
venv:
	@echo '# Preparing development environment...'
	@echo '...cloning .env...'
	@cp -n ./config/.env.example ./.env || :
	@echo '...preparing git hooks...'
	@cp -n ./config/pre-push ./.git/hooks/pre-push || :
	@echo '...installing uv...'
	@curl -LsSf https://github.com/astral-sh/uv/releases/download/$(UV_VERSION)/uv-installer.sh | sh
	@echo '...preparing venv...'
	@$(PYTHON_ALIAS) -m venv .venv --prompt="VENV[DEV]" --clear --upgrade-deps
	@. ./.venv/bin/activate && pip install --upgrade pip && uv pip install --editable $(INSTALL_OPTIONS) --constraint constraints
	@echo '...development environment ready! Activate venv using `. ./.venv/bin/activate`.'

# Sync environment with uv based on constraints
sync:
	@echo '# Synchronizing dependencies...'
	@$(if $(findstring $(UV_VERSION), $(shell uv --version | head -n1 | cut -d" " -f2)), , @echo '...updating uv...' && curl -LsSf https://github.com/astral-sh/uv/releases/download/$(UV_VERSION)/uv-installer.sh | sh)
	@uv pip install --editable $(INSTALL_OPTIONS) --constraint constraints
	@echo '...finished!'

# Generate a set of locked dependencies from pyproject.toml
lock:
	@echo '# Locking dependencies...'
	@uv pip compile pyproject.toml -o constraints --all-extras
	@echo '...finished!'

# Update and lock dependencies from pyproject.toml
update:
	@echo '# Updating dependencies...'
	@$(if $(findstring $(UV_VERSION), $(shell uv --version | head -n1 | cut -d" " -f2)), , @echo '...updating uv...' && curl -LsSf https://github.com/astral-sh/uv/releases/download/$(UV_VERSION)/uv-installer.sh | sh)
	@uv --no-cache pip compile pyproject.toml -o constraints --all-extras --upgrade
	@uv pip install --editable $(INSTALL_OPTIONS) --constraint constraints
	@echo '...finished!'

# Run formatter.
format:
	@ruff check --quiet --fix $(SOURCES_PATH) $(TESTS_PATH)
	@ruff format --quiet $(SOURCES_PATH) $(TESTS_PATH)

# Run linters and code checks.
lint:
	@bandit -r $(SOURCES_PATH)
	@ruff check $(SOURCES_PATH) $(TESTS_PATH)
	@pyright --project ./

# Run tests suite.
test:
	@$(PYTHON_ALIAS) -B -m pytest -vv --cov=$(SOURCES_PATH) --rootdir=$(TESTS_PATH)

release: lint test
	@echo '# Preparing release...'
	@python -m build && python -m twine upload --skip-existing dist/*
	@echo '...finished!'
