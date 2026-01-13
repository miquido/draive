# Draive

[![PyPI](https://img.shields.io/pypi/v/draive)](https://pypi.org/project/draive/)
![Python Version](https://img.shields.io/badge/Python-3.13+-blue)
[![License](https://img.shields.io/github/license/miquido/draive)](https://github.com/miquido/draive/blob/main/LICENSE)

Draive is a typed Python framework for building GenAI applications with strict state modeling,
scoped context management, and composable async workflows.

It is designed for teams that want predictable, testable AI systems instead of prompt scripts with
hidden global state.

## Why Draive

- Explicitly typed API surface built on `State`, with runtime validation and JSON schema support.
- Unified generation interfaces:
    `TextGeneration`, `ModelGeneration`, `ImageGeneration`, and `AudioGeneration`.
- Built-in tool orchestration through `@tool` and `Toolbox` with model-managed tool turns.
- Shared multimodal model for text, resources, and artifacts across generation and retrieval flows.
- First-class evaluation and guardrails integrated in the same context/state model.
- Provider adapters for OpenAI, Anthropic, Gemini, Mistral, Cohere, Bedrock, Ollama, and vLLM.

## Architecture At A Glance

- `ctx.scope(...)` defines an execution boundary and binds state/disposables.
- `draive.generation` provides high-level typed generation facades.
- `draive.steps` provides composable pipeline execution (`Step`, `StepState`).
- `draive.multimodal` standardizes text/resources/artifacts/tagged content.
- `draive.embedding` and `VectorIndex` power retrieval workloads.
- `draive.evaluation` and `draive.evaluators` power quality verification.

## Quick Start

### Installation

```bash
pip install "draive[openai]"
```

Create `.env` and provide your provider key:

```env
OPENAI_API_KEY=your-api-key
```

### Minimal Text Generation

The snippet below shows the core execution pattern used throughout Draive:

- load environment variables once,
- open a scoped context with provider config and client disposable,
- call generation API from inside that scope.

```python
import asyncio

from draive import TextGeneration, ctx, load_env
from draive.openai import OpenAI, OpenAIResponsesConfig

load_env()


async def main() -> None:
    async with ctx.scope(
        "quickstart",
        OpenAIResponsesConfig(model="gpt-5-mini"),
        disposables=(OpenAI(),),
    ):
        result: str = await TextGeneration.generate(
            instructions="You are a helpful assistant",
            input="Give me three taglines for an AI travel app.",
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

How this works:

- `ctx.scope("quickstart", ...)` opens an execution scope used for tracing, state lookup, and
    dependency lifetime.
- `OpenAIResponsesConfig(...)` selects model and generation defaults for this scope.
- `disposables=(OpenAI(),)` registers the provider client and closes it automatically.
- `TextGeneration.generate(...)` resolves active provider + model from context and returns `str`.

### Typed Structured Generation

`ModelGeneration` turns model output into a typed `State`. This is useful when downstream code needs
strong contracts instead of plain text.

```python
from collections.abc import Sequence

from draive import ModelGeneration, State, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


class PersonInfo(State, serializable=True):
    name: str
    role: str
    skills: Sequence[str]


async with ctx.scope(
    "typed-extraction",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    person: PersonInfo = await ModelGeneration.generate(
        PersonInfo,
        instructions="Extract person details from the sentence.",
        input="Ava is a backend engineer experienced in Python and Postgres.",
        schema_injection="simplified",
    )
```

Notes:

- `serializable=True` is required for schema-based decoding into your `State`.
- `schema_injection="simplified"` appends a compact schema description to your instructions.
- Returned value is a validated `PersonInfo`, not raw JSON.

### Tool Use In One Scope

Tools are regular async functions decorated with `@tool`. They can be passed to generation calls and
invoked by the model when appropriate.

```python
from datetime import datetime, timezone

from draive import TextGeneration, ctx, tool
from draive.openai import OpenAI, OpenAIResponsesConfig


@tool(description="Returns current UTC timestamp in ISO-8601 format")
async def current_utc_time() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


async with ctx.scope(
    "tools-demo",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    reply: str = await TextGeneration.generate(
        instructions="Use available tools when they improve accuracy.",
        input="What is the current UTC time?",
        tools=[current_utc_time],
    )
```

This lets you keep model reasoning and side-effectful capabilities separated and typed.

### Step Pipelines (Refactor-Aligned)

The refactored pipeline abstraction is `Step` + `StepState` (from `draive.steps`). Use it when
single-call generation is no longer enough and you need multi-stage flows.

Common patterns include:

- sequential orchestration (`Step.sequence(...)`),
- guarded loops (`Step.loop(...)`),
- explicit state preservation/restoration across phases.

For complete walkthroughs, see the dedicated step guides.

## Where Next

- [Getting started](./getting-started/index.md)
- [Guides](./guides/BasicUsage.md)
- [Cookbooks](./cookbooks/BasicRAG.md)
