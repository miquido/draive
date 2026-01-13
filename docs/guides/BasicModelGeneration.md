# Basic Model Generation

Use `ModelGeneration.generate(...)` to produce typed `State` objects from model output.

This API is most useful when generated content feeds business logic, storage, or API responses that
require explicit shape guarantees.

## Prerequisites

- Python 3.13+
- OpenAI credentials loaded via `load_env()`

```python
from draive import load_env

# Reads local .env into process environment.
load_env()
```

## 1. Define Output Schema As Serializable State

`serializable=True` is required for schema-driven decoding.

```python
from draive import State


class InterestingPlace(State, serializable=True):
    name: str
    description: str | None = None
```

## 2. Configure Provider In `ctx.scope`

Model generation resolves provider state from the active scope.

```python
from draive import ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


async with ctx.scope(
    "basic_generation",
    # Generation parameters resolved by statemethods.
    OpenAIResponsesConfig(model="gpt-5-mini"),
    # Client lifecycle managed by context scope.
    disposables=(OpenAI(),),
):
    ...
```

## 3. Generate Typed Output

```python
from draive import ModelGeneration


async with ctx.scope(
    "basic_generation",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    place: InterestingPlace = await ModelGeneration.generate(
        InterestingPlace,
        instructions="You are a helpful travel assistant.",
        input="Recommend one must-see location in London.",
    )

    print(place)
```

The return value is already validated and typed as `InterestingPlace`.

## Useful Options

- `schema_injection="full" | "simplified" | "skip"`
- `tools=[...]` or `tools=Toolbox.of(...)` to enable tool use during generation
- `examples=[(input, expected_state), ...]` for few-shot structured generation
- `decoder=...` to override default decoding behavior

## Next Steps

- Chain generation into `Step` pipelines.
- Add retrieval-backed tool calls.
- Add regression quality checks with `draive.evaluation` suites.
