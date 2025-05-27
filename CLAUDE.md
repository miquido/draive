# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
- `make venv` - Setup development environment and install git hooks
- `make sync` - Sync dependencies with uv lock file
- `make update` - Update and lock dependencies

### Code Quality
- `source .venv/bin/activate && make format` - Format code with Ruff (line length 100)
- `source .venv/bin/activate && make lint` - Run linters (Ruff + Bandit + Pyright strict mode)
- `source .venv/bin/activate && make test` - Run pytest with coverage
- `source .venv/bin/activate && pytest tests/test_specific.py` - Run single test file
- `source .venv/bin/activate && pytest tests/test_specific.py::test_function` - Run specific test

### Package Management
- `uv add <package>` - Add dependency
- `uv remove <package>` - Remove dependency

## Architecture Overview

**draive** is a Python framework for LLM applications built on haiway for state management.

### Core Concepts

**State Management**: Uses haiway's immutable State objects and ctx for dependency injection. All configuration and context flows through state scoping.

**Provider Abstraction**: Unified `LMM` interface across providers (OpenAI, Anthropic, Gemini, Mistral, Ollama, Bedrock). Each provider implements standardized `LMMContext`, `LMMInput`, `LMMCompletion` types.

**Multimodal Content**: `MultimodalContent` handles text, images, and media uniformly. Content elements are composable and convertible across formats.

### Key Components

**Stages**: Core processing pipeline units that transform `(LMMContext, MultimodalContent)`. Support composition, looping, caching, retry. Examples: `Stage.completion()`, `Stage.sequence()`, `Stage.loop()`.

**Tools**: Function calling via `@tool` decorator. `Toolbox` manages collections. Automatic schema generation and validation.

**Agents**: Stateful conversation entities with memory. `AgentWorkflow` for multi-agent systems.

**Generation APIs**: High-level interfaces like `TextGeneration`, `ModelGeneration` that handle complexity while allowing customization.

### Component Flow
```
Generation APIs → Stages → LMM → Provider Implementation
```

### Patterns

1. **Immutable State**: All state objects are frozen
2. **Context Scoping**: Configuration flows through execution contexts
3. **Async First**: Fully asynchronous throughout
4. **Composability**: Small focused components combine into complex workflows
5. **Type Safety**: Heavy use of generics and protocols

### Entry Points

- Simple: `TextGeneration.generate()`, `ModelGeneration.generate()`
- Tools: `@tool` decorator with `tools` parameter
- Complex: `Stage` composition and `Agent` systems
- Setup: Context managers with provider configs

### Testing

Uses pytest with async support. Tests are in `tests/` directory. Key test patterns:
- Mock LMM responses for unit tests
- Use actual providers for integration tests
- Test both sync and async code paths
