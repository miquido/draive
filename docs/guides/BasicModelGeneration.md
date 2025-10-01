# Basic Model Generation Guide

Use Draive's model generation utilities to turn natural-language prompts into strongly typed Python objects. This guide walks through the minimal setup, explains how each piece fits together, and highlights a few customisation knobs you can try right away.

## Prerequisites
- Python 3.12+ with Draive installed and activated in the project virtual environment (`.venv`).
- An `.env` file containing a valid `OPENAI_API_KEY`. The key stays local; Draive loads it at runtime.
- Familiarity with Python's async/await syntax. All generation APIs are asynchronous.

> Tip: `print(...)` is perfectly fine for quick experiments. When you integrate the code into an application prefer `ctx.log_info(...)`, `ctx.log_warn(...)`, etc., so your logs flow through Haiway's observability stack.

## 1. Load Provider Credentials
Call `load_env()` once during startup to populate environment variables from `.env`. Draive looks up provider credentials (such as `OPENAI_API_KEY`) from the process environment.

```python
from draive import load_env

load_env()
```

If you already manage secrets elsewhere (Docker, cloud secret manager, CI), skip this step and make sure the variables are present before launching your code.

## 2. Describe the Structured Response
`ModelGeneration` returns instances of a `DataModel`. Define the schema with standard Python type hints so you get validation, IDE auto-complete, and predictable shapes back from the model.

```python
from draive import DataModel


class InterestingPlace(DataModel):
    """Structured description of a sight we want to surface to the user."""

    name: str
    description: str | None = None
```

Every attribute becomes part of the contract with the model. Optional fields should default to `None` for clarity.

## 3. Prepare the Haiway Context Scope
All provider calls run inside a Haiway `ctx.scope` so dependencies are tracked and disposed correctly. In this scope you pass:
- A unique label (for tracing/logging).
- Provider configuration (here `OpenAIResponsesConfig`).
- Disposable dependencies such as the `OpenAI` client instance.

```python
from draive import ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


responses_config = OpenAIResponsesConfig(model="gpt-4o-mini")

async with ctx.scope(
    "basic_generation",  # label shown in logs/metrics
    responses_config,
    disposables=(OpenAI(),),
):
    ...
```

You can place additional state objects inside the scope (feature flags, caches, etc.) whenever the generation step needs them.

## 4. Generate a Typed Object
With the scope active, call `ModelGeneration.generate(...)`. Provide the `DataModel` class, high-level instructions, and the user input that should drive the generation.

```python
from draive import ModelGeneration


async with ctx.scope(
    "basic_generation",
    OpenAIResponsesConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
):
    place: InterestingPlace = await ModelGeneration.generate(
        InterestingPlace,
        instructions="You are a helpful travel assistant.",
        input="Recommend one must-see location in London",
    )

    ctx.log_info(f"Generated place: {place}")
```

Behind the scenes Draive crafts a prompt that asks OpenAI's structured response API to honour your schema. The result is validated before being returned, so `place.name` and `place.description` are always present with the correct types.

Example output:

```text
InterestingPlace(name='The British Museum', description='A world-famous museum dedicated to human history, art, and culture.')
```

## Customise the Call
- **Model choice:** Swap `gpt-4o-mini` for any supported JSON mode model (`gpt-4o`, `gpt-4.1-mini`, etc.) via `OpenAIResponsesConfig`.
- **System behaviour:** Adjust `instructions` to inject tone, guardrails, or extra context such as "Respond in 2 short sentences".
- **User input:** Pass any serialisable payload. For complex prompts consider building a `MultimodalContent` with references, images, and text snippets.
- **Validation strategy:** Add `validators=` to `ModelGeneration.generate` if you want to run post-generation checks (see the evaluation guide).

## Where to Go Next
- Return multiple locations by switching to a `DataModel` that wraps a `Sequence[InterestingPlace]`.
- Chain the generation result into downstream stages (for example, rendering a personalised itinerary) by leaving the scope open and calling additional async helpers.
- Explore `draive/models/tools` if you need the model to call tools or use retrieval augmentation while producing structured data.
