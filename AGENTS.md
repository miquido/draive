Draive is a python framework helping to build high-quality Gen-AI applications. It focuses on strict typing and functional programming principles extended with structured concurrency concepts. It delivers opinionated, strict rules and patterns resulting in modular, safe and highly maintainable applications.

## Development Toolchain

- Python: 3.14+
- Virtualenv: managed by uv, available at `./.venv`, assume already set up and working within venv
- Formatting: Ruff formatter (`make format`), no other formatter
- Linters/Type-checkers: Ruff, Bandit, Pyright (strict). Run via `make lint`
- Tests: pytest, run via `make test` or targeted `pytest tests/test_...::test_...`

## Project Layout

Top-level code lives under `src/draive/`, with key packages:

- `draive/models/` — core model abstractions (`GenerativeModel`, context/input/output types, session and tool-related model types)
- `draive/tools/` — tool abstractions and orchestration (`Tool`, `FunctionTool`, `Toolbox`, providers)
- `draive/agents/` — lightweight async agent wrappers and delegation (`Agent`, `AgentsGroup`, agent message/identity types)
- `draive/generation/` — typed generation facades (`text/`, `image/`, `audio/`, `model/`) with `state.py`, `types.py`, and `default.py`
- `draive/conversation/` — higher-level chat/realtime flows (`completion/`, `realtime/`)
- `draive/multimodal/` — multimodal content and templates (`MultimodalContent`, `TextContent`, `ArtifactContent`, template helpers)
- `draive/resources/` — resource references, fetching/uploading interfaces, repository abstractions
- `draive/embedding/` — embeddings, similarity/search/mmr, and typed embedding/vector index state
- `draive/guardrails/` — moderation, privacy, quality, and safety guardrail states/types
- `draive/steps/` — pipeline step abstractions (`Step`, `StepState`, composition/execution helpers)
- `draive/evaluation/` — evaluation primitives (evaluators, scenarios, suites, scores)
- `draive/evaluators/` — ready-to-use evaluator catalog (coherence, relevance, safety, jailbreak, etc.)
- `draive/splitters/` — text splitting helpers
- `draive/helpers/` — high-level utilities (instruction preparation/refinement, volatile vector index)
- `draive/utils/` — shared low-level utility helpers
- Provider adapters (feature-specific modules per provider):
  - `draive/openai/`, `draive/anthropic/`, `draive/mistral/`, `draive/gemini/`, `draive/vllm/`, `draive/ollama/`, `draive/bedrock/`, `draive/cohere/`
- Integrations (opt-in extras):
  - `draive/httpx/`, `draive/aws/`, `draive/qdrant/`, `draive/mcp/`, `draive/postgres/`, `draive/opentelemetry/`, `draive/rabbitmq/`

Public exports are centralized in `src/draive/__init__.py`.

## Style & Patterns

- Draive is built on top of Haiway (state, context, observability, config). See: [Haiway](https://github.com/miquido/haiway)
- Import symbols from `draive` directly when possible: `from draive import State, ctx`
- Use context scoping (`ctx.scope(...)`) to bind scoped `Disposables`, active `State` instances, and avoid global state

### Typing & Immutability

- Use strict typing and modern Python 3.14 syntax
- No untyped public APIs; avoid loose `Any` except at explicit third-party boundaries
- Prefer explicit attribute access with static types; avoid dynamic `getattr` except at narrow boundaries
- Prefer abstract immutable protocols (`Mapping`, `Sequence`, `Iterable`) over concrete mutable collections in public APIs
- Use `final` where applicable; avoid inheritance where composition is clearer
- Use precise unions (`|`) and narrow with `match`/`isinstance`; avoid `cast` unless provably safe and localized
- Favor structural typing (Protocols) for async clients and adapters
- Guard immutability at context boundaries; errors should aid debugging without leaking sensitive data

### Concurrency & Async

- All I/O is async; keep boundaries async
- Use context task helpers (`ctx.spawn`, `ctx.spawn_background`) for concurrent/detached work where appropriate
- Rely on Haiway/asyncio coroutine patterns; avoid custom threading
- Never block the event loop with synchronous long-running operations

### Exceptions & Error Translation

- Translate provider/SDK errors into typed domain exceptions
- Don’t raise bare `Exception`; preserve meaningful context
- Wrap third-party exceptions at boundaries and include actionable context (`provider`, `operation`, identifiers) while redacting sensitive payloads

### Logging & Observability

- Use `ctx` observability helpers (`ctx.log_*`, `ctx.record`) instead of `print`/`logging`
- Surface user-facing failures via structured events before raising typed exceptions

## Testing & CI

- No network in unit tests; mock providers/HTTP boundaries
- Keep tests fast and specific to changed code
- Use fixtures from `tests/` or add focused ones; avoid heavy integration scaffolding for unit coverage
- Linting/type gates: `make format` then `make lint`
- Mirror package layout in `tests/`; prefer parametrization over loops
- Test async flows with `pytest.mark.asyncio`; use `ctx.scope` in tests to isolate state
- Keep `reveal_type`-style type assertions local and remove before committing

### Self-verification

- Ensure strict type-checking soundness as part of the workflow
- Don’t silence errors; resolve root causes
- Verify correctness with focused unit tests or targeted ad-hoc scripts
- Capture tricky edge cases with regression tests before fixes

## Documentation

- Public symbols: add NumPy-style docstrings with `Parameters`, `Returns`, and `Raises`
- Internal/private helpers: prefer self-explanatory naming over docstrings
- If behavior/API changes, update relevant docs in `docs/` and examples
- Skip module docstrings
- Include usage snippets that exercise async scopes and state wiring through `ctx`

### Docs (MkDocs)

- Site is built with MkDocs + Material; PlantUML diagrams are built via `mkdocs-build-plantuml-plugin`
- Register navigation in `mkdocs.yml` (`nav:` section)
- Lint docs with `make docs-lint` and format with `make docs-format` after edits
- Keep docstrings high-quality and aligned with public APIs

## Security & Secrets

- Never log secrets or full request bodies containing keys/tokens
- Use environment variables for credentials, resolve via helpers like `getenv_str`

## Contribution Checklist

- Build: `make format` succeeds
- Quality: `make lint` is clean (Ruff, Bandit, Pyright strict)
- Tests: `make test` passes; add/update tests if behavior changes
- Types: strict, no ignores, no loosening of typing
- API surface: update `__init__.py` exports and docs when public surface changes
