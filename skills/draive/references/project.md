# Project structure and tooling

Draive does not enforce a layout, but the Haiway guidelines (Haiway docs: *Organizing Packages*)
make the dependency direction explicit. Follow them in new projects and keep an existing project's
layout if it already has one.

## Package layout

```
pyproject.toml
.python-version          # 3.15
Makefile
src/
├── entrypoint_a/        # CLI, HTTP server, worker - outermost layer
│   ├── __init__.py      # usually empty
│   ├── __main__.py
│   └── config.py
├── features/            # high level capabilities, orchestrate solutions
│   └── feature_a/
├── solutions/           # project specific building blocks, hide integrations
│   └── solution_a/
├── integrations/        # one third party service or resource per package
│   └── integration_a/
└── commons/             # shared types, errors, utilities - depends on nothing
tests/                   # mirrors src/
```

Dependency direction, from the outside in: entrypoints -> features -> solutions -> integrations ->
commons.

- Entrypoints are depended on by nobody; several are fine and keep runtimes separate.
- Features depend on solutions (and commons), **not on integrations**.
- Solutions never depend on features or entrypoints; they wrap integrations behind project types.
- An integration depends on commons only.
- `commons` depends on no other package of the project; keep it small. Break a circular
  dependency inside one level by merging at most three linked packages into a containing package.
- The top-level `features`, `solutions` and `integrations` packages export nothing - they only
  group.

## Inside a feature, solution or integration

```
feature_a/
├── __init__.py          # exports public symbols only (usually state and types)
├── types.py             # Protocols, payload States, errors - declarations, no logic, no imports
│                        #   from the rest of the package
├── config.py            # constants and environment configuration - depends on types.py only
├── state.py             # service State(s) + @statemethod entry points; may default a field to an
│                        #   implementation
└── implementation.py    # private functions behind the Protocols + the factory building the State
                         #   (name it after the variant when there are several: files.py, s3.py)
```

`__init__.py` exports the `State`, the types and the factories - never the implementation
functions. Other packages use those and receive the implementation through the context.

## pyproject.toml

Draive requires Python 3.15 (there are no wheels for the free-threaded build). Declare the integrations
you use as extras of `draive`; keep tooling in dependency groups, never published:

```toml
[project]
name = "my-app"
requires-python = ">=3.15"
dependencies = ["draive[openai,httpx,postgres]"]   # extras per integration, pinned to a compatible range

[dependency-groups]
dev = ["bandit", "pyright", "pytest", "pytest-asyncio", "pytest-cov", "ruff"]

[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_default_fixture_loop_scope = "function"   # fresh event loop per test, avoids a warning

[tool.ruff]
target-version = "py315"
line-length = 100
lint.select = ["E", "F", "A", "I", "B", "PL", "W", "C", "RUF", "UP"]

[tool.pyright]
pythonVersion = "3.15"
include = ["./src"]
typeCheckingMode = "strict"
reportUnnecessaryTypeIgnoreComment = "error"
```

Pin the interpreter with a `.python-version` file and manage the environment with `uv`
(`uv sync`, `uv run ...`). `uv sync` removes what is not declared, so declare every extra the code
imports. Draive depends on Haiway, which ships a compiled extension; wheels exist for Linux
(x86_64, aarch64), macOS (arm64) and Windows (amd64), installing from source needs a C compiler.

## Makefile / commands

Expose the checks as one-word targets so humans and agents run the same thing:

```make
format:
	@uv run ruff check --quiet --fix src tests
	@uv run ruff format --quiet src tests

lint:
	@uv run bandit -r src
	@uv run ruff check src tests
	@uv run pyright --project ./

test:
	@uv run python -B -m pytest -v --cov=src
```

Definition of done for a change: `format`, `lint` and `test` all clean. Fix type errors rather than
adding `# type: ignore`, `cast` or `Any`; do not widen ruff ignores to make a finding disappear.

## Tests

Mirror the package layout in `tests/`. Add an autouse fixture so no task leaks between tests:

```python
@pytest.fixture(autouse=True)
def cleanup_background_tasks() -> Generator[None]:
    yield
    ctx.shutdown_background_tasks()
```

Capture an edge case as a regression test before fixing it. How to fake dependencies is in
`patterns.md` and `recipes.md`.
