Draive is a python framework helping to build high-quality Gen-AI applications. Focuses on strict typing and functional programming principles extended with structured concurrency concepts. Delivers opinionated, strict rules and patterns resulting in modular, safe and highly maintainable applications.

## Development Toolchain

- Python: 3.13+
- Virtualenv: managed by uv, available at `./.venv`, assume already set up and working within venv
- Formatting: Ruff formatter (`make format`), no other formatter
- Linters/Type-checkers: Ruff, Bandit, Pyright (strict). Run via `make lint`
- Tests: using pytest, run using `make test` or targeted `pytest tests/test_...::test_...`

## Project Layout

Top-level code lives under `src/draive/`, key packages and what they contain:

- `draive/models/` — core model abstractions: `GenerativeModel`, tools (`models/tools`), instructions handling
- `draive/generation/` — typed generation facades for text, image, audio, and model wiring (`state.py`, `types.py`, `default.py`)
- `draive/conversation/` — higher-level chat/realtime conversations (completion and realtime sessions)
- `draive/multimodal/` — content types and helpers: `MultimodalContent`, `TextContent`, `ArtifactContent`
- `draive/resources/` — resource references and blobs: `ResourceContent`, `ResourceReference`, repository interfaces
- `draive/parameters/` — strongly-typed parameter schemas and validation helpers
- `draive/embedding/` — vector operations, similarity, indexing, and typed embedding states
- `draive/guardrails/` — moderation, privacy, quality verification states and types
- `draive/stages/` — pipeline stages abstractions and helpers
- `draive/utils/` — utilities (e.g., `Memory`, `VectorIndex`)
- Provider adapters — unified shape per provider:
  - `draive/openai/`, `draive/anthropic/`, `draive/mistral/`, `draive/gemini/`, `draive/vllm/`, `draive/ollama/`, `draive/bedrock/`, `draive/cohere/`
  - Each has `config.py`, `client.py`, `api.py`, and feature-specific modules.
- Integrations: `draive/httpx/`, `draive/aws/`, `draive/qdrant/`, `draive/mcp/`, `draive/postgres/`, `draive/opentelemetry/` (opt-in extras)

Public exports are centralized in `src/draive/__init__.py`.

## Style & Patterns

- Draive is built on top of Haiway (state, context, observability, config). See: [Haiway](https://github.com/miquido/haiway)
- Import symbols from `draive` directly: `from draive import State, ctx`
- Use context scoping (`ctx.scope(...)`) to bind scoped `Disposables`, active `State` instances and avoid global state

### Typing & Immutability

- Ensure latest, most strict typing syntax available from python 3.13+
- Strict typing only: no untyped public APIs, no loose `Any` unless required by third-party boundaries
- Prefer explicit attribute access with static types. Avoid dynamic `getattr` except at narrow boundaries.
- Prefer abstract immutable protocols: `Mapping`, `Sequence`, `Iterable` over `dict`/`list`/`set` in public types
- Use `final` where applicable; avoid inheritance, prefer type composition
- Use precise unions (`|`) and narrow with `match`/`isinstance`, avoid `cast` unless provably safe and localized
- Favor structural typing (Protocols) for async clients and adapters; runtime-checkable protocols like `HTTPRequesting` keep boundaries explicit.
- Guard immutability with assertions when crossing context boundaries; failure messages should aid debugging but never leak secrets.

### Concurrency & Async

- All I/O is async, keep boundaries async and use `ctx.spawn` for detached tasks
- Ensure structured concurrency concepts and valid coroutine usage
- Rely on haiway and asyncio packages with coroutines, avoid custom threading
- Await long-running operations directly; never block the event loop with sync calls.

### Exceptions & Error Translation

- Translate provider/SDK errors into appropriate typed exceptions
- Don’t raise bare `Exception`, preserve contextual information in exception construction
- Wrap third-party exceptions at the boundary and include actionable context (`provider`, `operation`, identifiers) while redacting sensitive payloads.

### Logging & Observability

- Use observability hooks (logs, metrics, traces) from `ctx` helper (`ctx.log_*`, `ctx.record`) instead of `print`/`logging`—tests assert on emitted events.
- Surface user-facing errors via structured events before raising typed exceptions.

## Testing & CI

- No network in unit tests, mock providers/HTTP
- Keep tests fast and specific to the code you change, start with unit tests around new types/functions and adapters
- Use fixtures from `tests/` or add focused ones; avoid heavy integration scaffolding
- Linting/type gates must be clean: `make format` then `make lint`
- Mirror package layout in `tests/`; colocate new tests alongside features and prefer `pytest` parametrization over loops.
- Test async flows with `pytest.mark.asyncio`; use `ctx.scope` in tests to isolate state and avoid leaking globals.
- Use `pyright`-style type assertions (e.g., `reveal_type`) only locally and delete them before committing.

### Self-verification

- Ensure type checking soundness as a part of the workflow
- Do not mute or ignore errors, double-check correctness and seek for solutions
- Verify code correctness with unit tests or by running ad-hoc scripts
- Capture tricky edge cases in regression tests before fixing them to prevent silent behaviour changes.

## Documentation

- Public symbols: add NumPy-style docstrings. Include Parameters/Returns/Raises sections and rationale
- Internal and private helpers: avoid docstrings, keep names self-explanatory
- If behavior/API changes, update relevant docs under `docs/` and examples if applicable
- Skip module docstrings
- Add usage snippets that exercise async scopes; readers should see how to wire states through `ctx`.

### Docs (MkDocs)

- Site is built with MkDocs + Material and `mkdocstrings` for API docs.
- Author pages under `docs/` and register navigation in `mkdocs.yml` (`nav:` section).
- Lint `make docs-lint` and format `make docs-format` after editing.
- Keep docstrings high‑quality; `mkdocstrings` pulls them into reference pages.
- When adding public APIs, update examples/guides as needed and ensure cross-links render.

## Security & Secrets

- Never log secrets or full request bodies containing keys/tokens
- Use environment variables for credentials, resolve via helpers like `getenv_str`

## Contribution Checklist

- Build: `make format` succeeds
- Quality: `make lint` is clean (Ruff, Bandit, Pyright strict)
- Tests: `make test` passes, add/update tests if behavior changes
- Types: strict, no ignores, no loosening of typing
- API surface: update `__init__.py` exports and docs if needed
