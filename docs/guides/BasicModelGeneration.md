# Basic Model Generation

Use `ModelGeneration.generate(...)` to produce typed `State` objects from model output.

This API is most useful when generated content feeds business logic, storage, or API responses that
require explicit shape guarantees.

## Prerequisites

- Python 3.14+
- OpenAI credentials loaded via `load_env()`

```python
from draive import load_env

# Reads local .env into process environment.
load_env()
```

## 1. Define Output Schema As Serializable State

`serializable=True` makes the JSON schema requirement explicit - the class fails to define when
any of its fields can't be represented in the schema used for decoding.

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
    OpenAIResponsesConfig(model="gpt-5.5"),
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
    OpenAIResponsesConfig(model="gpt-5.5"),
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

## Schema Enforcement

Providers enforce the requested schema whenever they can. OpenAI responses request their
strict structured output mode, which is what makes the API - rather than the prompt - guarantee
the shape, so decoding cannot fail on a well formed response.

Strict mode demands a value for every field, so a field with a default is filled by the model
instead of being left out. Declare a field as optional (`str | None = None`) when "not present"
is a meaningful answer, and the model can report it as `null`.

A few shapes have no strict equivalent and fall back to delivering the schema as a hint, where
the model may answer with content that fails decoding:

- `Any` valued fields, including `Meta` and `Mapping[str, Any]`
- `tuple[...]` fields and unparameterized sequences
- self-referencing (recursive) models
- models nested more than ten objects deep

`Mapping[str, str]` and other fully typed mappings do stay strict. They are the one exception to
"every field is filled" - the API keeps them optional, so the model may leave one out entirely.

## Useful Options

- `schema_injection="full" | "simplified" | "skip"`
- `tools=[...]` or `tools=Toolbox.of(...)` to enable tool use during generation
- `examples=[(input, expected_state), ...]` for few-shot structured generation
- `decoder=...` to override default decoding behavior

## Next Steps

- Chain generation into `Step` pipelines.
- Add retrieval-backed tool calls.
- Add regression quality checks with `draive.evaluation` suites.
