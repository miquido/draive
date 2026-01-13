# Basics

This guide covers Draive fundamentals: typed `State`, scoped context, immutable updates, and
observability.

## Typed State

`State` is the main model primitive used across configuration, runtime artifacts, and generated
structures.

```python
from draive import State


class BasicState(State):
    identifier: str
    value: int
```

Instances are immutable. Use `.updating(...)` to create modified copies.

```python
basic_state = BasicState(identifier="basic", value=42)
updated_state = basic_state.updating(value=21)
```

## Serializable State

Use `serializable=True` when you need JSON conversion, schema generation, and structured decoding.

```python
from collections.abc import Sequence

from draive import State


class BasicModel(State, serializable=True):
    username: str
    tags: Sequence[str] | None = None


payload = BasicModel(username="John Doe", tags=("example", "json"))
print(payload.to_json(indent=2))
print(BasicModel.json_schema(indent=2))
```

## Context Scope

`ctx.scope(...)` binds state and services for one execution boundary.

```python
from draive import ctx


async with ctx.scope("basics", basic_state):
    # Resolve active state by type.
    current: BasicState = ctx.state(BasicState)
    print(current)
```

Use `ctx.updating(...)` for temporary overrides in nested scopes.

```python
async with ctx.scope("basics", basic_state):
    with ctx.updating(basic_state.updating(identifier="updated")):
        print(ctx.state(BasicState))
```

## Logging And Metrics

Use `ctx.log_*` and `ctx.record_*` for observability instead of `print` in production code.

```python
from draive import setup_logging
from haiway import LoggerObservability

setup_logging("basics")

async with ctx.scope("basics", observability=LoggerObservability()):
    ctx.log_info("structured log entry")
    ctx.record_info(attributes={"event": "example"})
```

Use `load_env()` during startup when credentials are stored in `.env`.
