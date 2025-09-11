# AGENTS.md

Rules for coding agents to contribute correctly and safely.

## Development Toolchain

- Python: 3.12
- Virtualenv: managed by uv, available at `./.venv`, assume already set up and working within venv
- Formatting: Ruff formatter (`make format`), no other formatter
- Linters/Type-checkers: Ruff, Bandit, Pyright (strict). Run via `make lint`
- Tests: `make test` or targeted `pytest tests/test_...::test_...`

## Framework Foundation (Haiway)

- Draive is built on top of Haiway (state, context, observability, config). See: https://github.com/miquido/haiway
- Import symbols from `haiway` directly: `from haiway import State, ctx`
- Use context scoping (`ctx.scope(...)`) to bind active `State` instances and avoid global state
- All logs go through `ctx.log_*`; do not use `print`.

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
- Integrations: `draive/httpx/`, `draive/mcp/`, `draive/opentelemetry/` (opt-in extras)

Public exports are centralized in `src/draive/__init__.py`.

## Style & Patterns

### Typing & Immutability

- Strict typing only: no untyped public APIs, no loose `Any` unless required by third-party boundaries
- Prefer explicit attribute access with static types. Avoid dynamic `getattr` except at narrow boundaries.
- Prefer abstract immutable protocols: `Mapping`, `Sequence`, `Iterable` over `dict`/`list`/`set` in public types
- Use `final` where applicable; avoid complex inheritance, prefer type composition
- Use precise unions (`|`) and narrow with `match`/`isinstance`, avoid `cast` unless provably safe and localized

### State & Context

- Use `haiway.State` for immutable data/config and service facades. Construct with classmethods like `of(...)` when ergonomic.
- Avoid in-place mutation; use `State.updated(...)`/functional builders to create new instances.
- Access active state through `haiway.ctx` inside async scopes (`ctx.scope(...)`).
- Public state methods that dispatch on the active instance should use `@statemethod` (see `GenerativeModel`).

#### Examples:

State with instantiation helper:
```python
from typing import Self
from haiway import State, statemethod, Meta, MetaValues

class Counter(State):
    @classmethod
    def of(
        cls,
        *,
        start: int = 0,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            value=start,
            meta=Meta.of(meta),
        )

    value: int
    meta: Meta
```

State with statemethod helper:
```python
from typing import Self, Protocol
from haiway import State, statemethod

class Printing(Protocol):
    async def __call__(self, text: str) -> None: ...

class Printer(State):
    @statemethod
    async def print(
        self,
        *,
        value: str | int | float,
    ) -> None:
        await self.printing(str(value))

    printing: Printing
```

### Observability & Logging

- Use `ctx.log_debug/info/warn/error` for logs; do not use `print`
- Log around generation calls, tool dispatch, provider requests/responses (without leaking secrets)
- Add appropriate metrics tracking using `ctx.record` where applicable
- Prefer structured/concise messages; avoid excessive verbosity in hot paths

### Concurrency & Async

- All I/O is async, keep boundaries async and use `ctx.spawn` for detached tasks
- Ensure structured concurrency concepts and valid coroutine usage
- Rely on haiway and asyncio packages with coroutines, avoid custom threading

### Exceptions & Error Translation

- Translate provider/SDK errors into appropriate typed exceptions
- Don’t raise bare `Exception`, preserve contextual information in exception construction

### Multimodal

- Build content with `MultimodalContent.of(...)` and prefer composing content blocks explicitly
- Use ResourceContent/Reference for media and data blobs
- Wrap custom types and data within ArtifactContent, use hidden when needed

## Testing & CI

- No network in unit tests, mock providers/HTTP
- Keep tests fast and specific to the code you change, start with unit tests around new types/functions and adapters
- Use fixtures from `tests/` or add focused ones; avoid heavy integration scaffolding
- Linting/type gates must be clean: `make format` then `make lint`

### Async tests

- Use `pytest-asyncio` for coroutine tests (`@pytest.mark.asyncio`).
- Prefer scoping with `ctx.scope(...)` and bind required `State` instances explicitly.
- Avoid real I/O and network; stub provider calls and HTTP.

#### Examples

```python
import pytest
from draive import State, ctx

class Example(State):
    name: str

@pytest.mark.asyncio
async def test_greeter_returns_greeting() -> None:
    async with ctx.scope(Example(name="Ada")):
        example: Example = ctx.state(Example)
        assert example.name == "Ada"
```

### Self verification

- Ensure type checking soundness as a part of the workflow
- Do not mute or ignore errors, double check correctness and seek for solutions
- Verify code correctness with unit tests or by running ad-hoc scripts
- Ask for additional guidance and confirmation when uncertain or about to modify additional elements

## Documentation

- Public symbols: add NumPy-style docstrings. Include Parameters/Returns/Raises sections and rationale when not obvious
- Internal helpers: avoid docstrings, keep names self-explanatory
- If behavior/API changes, update relevant docs under `docs/` and examples if applicable
- Skip module docstrings

### Docs (MkDocs)

- Site is built with MkDocs + Material and `mkdocstrings` for API docs.
- Author pages under `docs/` and register navigation in `mkdocs.yml` (`nav:` section).
- Preview locally: `make docs-server` (serves at http://127.0.0.1:8000).
- Build static site: `make docs` (outputs to `site/`).
- Keep docstrings high‑quality; `mkdocstrings` pulls them into reference pages.
- When adding public APIs, update examples/guides as needed and ensure cross‑links render.

## Security & Secrets

- Never log secrets or full request bodies containing keys/tokens
- Use environment variables for credentials, resolve via helpers like `getenv_str`

## Contribution Checklist

- Build: `make format` succeeds
- Quality: `make lint` is clean (Ruff, Bandit, Pyright strict)
- Tests: `make test` passes, add/update tests if behavior changes
- Types: strict, no ignores, no loosening of typing
- API surface: update `__init__.py` exports and docs if needed

## Code Search (ast-grep)

Use ast-grep for precise, structural search and refactors. Prefer `rg` for simple text matches.

- Invocation: `sg run --lang python --pattern "<PATTERN>" [FLAGS]`
- Useful patterns for this repo:
  - `ctx.scope($X)` — find context scopes
  - `@statemethod` — find state-dispatched classmethods
  - `class $NAME(State)` — locate `haiway.State` subclasses
  - `MultimodalContent.of($X)` — multimodal composition sites
  - `$FUNC(meta=$META)` — functions receiving `meta`
- Common flags:
  - `--json` — machine-readable results
  - `--selector field_declaration` — restrict matches (e.g., type fields)
  - `--rewrite "<REWRITE>"` — propose changes; always review diffs first

### Examples:

```bash
# Where do we open context scopes?
sg run --lang python --pattern "ctx.scope($X)" src

# Find all State types
sg run --lang python --pattern "class $NAME(State)" src

# Find multimodal builders
sg run --lang python --pattern "MultimodalContent.of($X)" src
```
