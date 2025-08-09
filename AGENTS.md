# AGENTS.md

Rules for coding agents to contribute correctly and safely.

## Development Toolchain

- Python: 3.12
- Virtualenv: managed by uv, available at `./.venv`
- Formatting: Ruff formatter (`make format`). No other formatter
- Linters/Type-checkers: Ruff, Bandit, Pyright (strict). Run via `make lint`
- Tests: `make test` or targeted `pytest tests/test_...::test_...`

## Style & Patterns

### Typing & Immutability

- Strict typing only: no untyped public APIs, no loose `Any` unless required by third-party boundaries
- Prefer abstract immutable protocols: `Mapping`, `Sequence`, `Iterable` over `dict`/`list`/`set` in public types
- Use `final` where applicable; avoid complex inheritance, prefer type composition
- Use precise unions (`|`) and narrow with `match`/`isinstance`, avoid `cast` unless provably safe and localized

### State & Context

- Use `haiway.State` for immutable data/config and service facades. Construct with classmethods like `of(...)` when ergonomic.
- Avoid in-place mutation; use `State.updated(...)`/functional builders to create new instances.
- Access active state through `haiway.ctx` inside async scopes (`ctx.scope(...)`).
- Public state methods that dispatch on the active instance should use `@statemethod` (see `GenerativeModel`).

### Observability & Logging

- Use `ctx.log_debug/info/warn/error` for logs; do not use `print`
- Log around generation calls, tool dispatch, provider requests/responses (without leaking secrets)
- Add appropriate metrics tracking using `ctx.record` where applicable
- Prefer structured/concise messages; avoid excessive verbosity in hot paths

### Concurrency & Async

- All I/O is async, keep boundaries async and use `ctx.spawn` for detached tasks
- Rely on asyncio package and coroutines, avoid custom threading

### Exceptions & Error Translation

- Translate provider/SDK errors into appropriate typed exceptions
- Don’t raise bare `Exception`, preserve provider/model identifiers in exception construction

### Multimodal

- Build content with `MultimodalContent.of(...)` and prefer composing content blocks explicitly.

## Testing & CI

- No network in unit tests, mock providers/HTTP
- Keep tests fast and specific to the code you change, start with unit tests around new types/functions and adapters
- Use fixtures from `tests/` or add focused ones; avoid heavy integration scaffolding
- Linting/type gates must be clean: `make format` then `make lint`

## Documentation

- Public symbols: add NumPy-style docstrings. Include Parameters/Returns/Raises sections and rationale when not obvious
- Internal helpers: avoid docstrings, keep names self-explanatory
- If behavior/API changes, update relevant docs under `docs/` and examples if applicable.

## Public API & Deprecations

- Export new public types/functions via `src/draive/__init__.py` and relevant package `__init__.py`.

## Security & Secrets

- Never log secrets or full request bodies containing keys/tokens
- Use environment variables for credentials, resolve via helpers like `getenv_str`

## Contribution Checklist

- Build: `make format` succeeds
- Quality: `make lint` is clean (Ruff, Bandit, Pyright strict)
- Tests: `make test` passes, add/update tests if behavior changes
- Types: strict, no ignores, no loosening of typing
- API surface: update `__init__.py` exports and docs if needed

## Extras

### Code Search (ast-grep)

- Use for precise AST queries and code search; prefer `rg` for simple text search
- Command: `sg run --lang python --pattern "<PATTERN>"`
- Patterns: `Meta.of($VAR)`, `meta: Meta`, `$FUNC(meta=$META)`, `class $NAME(Meta)`
- Flags: `--selector field_declaration`, `--json`, `--rewrite <FIX>` (review diffs)
