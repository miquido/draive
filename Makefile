SHELL := sh
.ONESHELL:
.SHELLFLAGS := -eu -c
.DELETE_ON_ERROR:

SOURCES_PATH := src
TESTS_PATH := tests

# load environment config from .env if able
-include .env

# a UV_PYTHON set in .env (e.g. UV_PYTHON=3.14) selects the interpreter for every uv command
ifdef UV_PYTHON
export UV_PYTHON
endif

ifndef UV_VERSION
	UV_VERSION := 0.13.7
endif

.PHONY: uv_check venv sync update format lint test docs docs-server docs-format docs-lint release


# Check installed UV version and install the pinned one if needed
uv_check:
	@echo 'Checking uv version...'

	# Install pinned version if not present
	@if ! command -v uv > /dev/null; then \
		echo '...installing uv $(UV_VERSION)...'; \
		curl -fLsS https://astral.sh/uv/$(UV_VERSION)/install.sh | sh; \
		if [ $$? -ne 0 ]; then \
			echo "...installing uv failed!"; \
			exit 1; \
		fi; \
	fi

	# Switch to the pinned version if a different one is installed
	@CURRENT_VERSION=$$(uv --version | head -n1 | cut -d" " -f2); \
	if [ "$$CURRENT_VERSION" != "$(UV_VERSION)" ]; then \
		echo "...switching uv $$CURRENT_VERSION to $(UV_VERSION)..."; \
		if ! uv self update $(UV_VERSION); then \
			INSTALLER=$$(curl -fLsS https://astral.sh/uv/$(UV_VERSION)/install.sh) \
				&& printf '%s' "$$INSTALLER" | sh \
				|| { echo "...updating uv failed!"; exit 1; }; \
		fi; \
		ACTIVE_VERSION=$$(uv --version | head -n1 | cut -d" " -f2); \
		if [ "$$ACTIVE_VERSION" != "$(UV_VERSION)" ]; then \
			echo "...uv $$ACTIVE_VERSION is still first on PATH, expected $(UV_VERSION)!"; \
			exit 1; \
		fi; \
	else \
		echo '...uv version is up-to-date!'; \
	fi

# Setup virtual environment for local development.
venv: uv_check
	@echo '# Preparing development environment...'
	@echo '...cloning .env...'
	@cp -n ./config/.env.example ./.env || :
	@echo '...preparing git hooks...'
	@cp -n ./config/pre-push ./.git/hooks/pre-push || :
	@echo '...preparing venv...'
	@uv sync --all-groups --all-extras --frozen --reinstall
	@echo '...development environment ready! Make targets and `uv run` use it without activation.'

# Sync environment with uv based on constraints
sync: uv_check
	@echo '# Synchronizing dependencies...'
	@uv sync --all-groups --all-extras --frozen
	@echo '...finished!'

# Update and lock dependencies from pyproject.toml
update: uv_check
	@echo '# Updating dependencies...'
	@uv sync --all-groups --all-extras --upgrade
	@echo '...finished!'

# Run formatter.
format:
	@uv run --frozen ruff check --quiet --fix $(SOURCES_PATH) $(TESTS_PATH)
	@uv run --frozen ruff format --quiet $(SOURCES_PATH) $(TESTS_PATH)

# Run linters and code checks.
lint:
	@uv run --frozen bandit -r $(SOURCES_PATH)
	@uv run --frozen ruff check $(SOURCES_PATH) $(TESTS_PATH)
	@uv run --frozen pyright --project ./

# Format Markdown docs and README
docs-format:
	@echo '# Formatting Markdown...'
	@uv run --frozen mdformat README.md docs
	@echo '...finished!'

# Lint Markdown docs and README
docs-lint:
	@echo '# Linting Markdown...'
	@uv run --frozen pymarkdown --config .pymarkdown.json scan README.md docs
	@echo '...finished!'

# Run tests suite.
test:
	@uv run --frozen python -B -m pytest -v --cov=$(SOURCES_PATH) --rootdir=$(TESTS_PATH)

# Build documentation for production
docs:
	@echo '# Building documentation...'
	@uv run --frozen mkdocs build --clean --strict
	@echo '...documentation built in site/ directory!'

# Build and serve documentation locally
docs-server:
	@echo '# Starting documentation server...'
	@uv run --frozen mkdocs serve --dev-addr 127.0.0.1:8000

release: lint test
	@echo '# Preparing release...'
	@uv build
	@uv publish
	@echo '...finished!'
