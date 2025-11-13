SHELL := sh
.ONESHELL:
.SHELLFLAGS := -eu -c
.DELETE_ON_ERROR:

SOURCES_PATH := src
TESTS_PATH := tests

# load environment config from .env if able
-include .env

ifndef UV_VERSION
	UV_VERSION := 0.9.9
endif

.PHONY: uv_check venv sync update format lint test docs docs-server docs-format docs-lint release

# Check installed UV version and install if needed
uv_check:
	@echo 'Checking uv version...'

	# Install if not present
	@if ! command -v uv > /dev/null; then \
		echo '...installing uv...'; \
		curl -fLsS https://astral.sh/uv/install.sh | sh; \
		if [ $$? -ne 0 ]; then \
			echo "...installing uv failed!"; \
			exit 1; \
		fi; \
	fi

	# Check version and update if needed
	@if command -v uv > /dev/null; then \
		CURRENT_VERSION=$$(uv --version | head -n1 | cut -d" " -f2); \
		if [ "$$(printf "%s\n%s" "$(UV_VERSION)" "$$CURRENT_VERSION" | sort -V | head -n1)" != "$(UV_VERSION)" ]; then \
			echo '...updating uv...'; \
			curl -fLsS https://astral.sh/uv/install.sh | sh; \
			if [ $$? -ne 0 ]; then \
				echo "...updating uv failed!"; \
				exit 1; \
			fi; \
		else \
			echo '...uv version is up-to-date!'; \
		fi; \
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
	@echo '...development environment ready! Activate venv using `. ./.venv/bin/activate`.'

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
	@ruff check --quiet --fix $(SOURCES_PATH) $(TESTS_PATH)
	@ruff format --quiet $(SOURCES_PATH) $(TESTS_PATH)

# Run linters and code checks.
lint:
	@bandit -r $(SOURCES_PATH)
	@ruff check $(SOURCES_PATH) $(TESTS_PATH)
	@pyright --project ./

# Format Markdown docs and README
docs-format:
	@echo '# Formatting Markdown...'
	@uv run mdformat --wrap 100 README.md docs
	@echo '...finished!'

# Lint Markdown docs and README
docs-lint:
	@echo '# Linting Markdown...'
	@uv run pymarkdown --config .pymarkdown.json scan README.md docs
	@echo '...finished!'

# Run tests suite.
test:
	@python -B -m pytest -v --cov=$(SOURCES_PATH) --rootdir=$(TESTS_PATH)

# Build documentation for production
docs:
	@echo '# Building documentation...'
	@mkdocs build --clean --strict
	@echo '...documentation built in site/ directory!'

# Build and serve documentation locally
docs-server:
	@echo '# Starting documentation server...'
	@mkdocs serve --dev-addr 127.0.0.1:8000

release: lint test
	@echo '# Preparing release...'
	@uv build
	@uv publish
	@echo '...finished!'
