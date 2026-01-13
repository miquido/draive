# First Steps

This guide walks through the core building blocks you will use in most Draive applications:
context scoping, typed state, generation, tools, retrieval, and guardrails.

## 1. Open A Context Scope

`ctx.scope(...)` is the runtime boundary for state resolution and disposable lifecycle.

```python
from draive import ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


async with ctx.scope(
    "app",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    # Inside this block, ctx.state(...) can resolve these states.
    ...
```

## 2. Define Typed State

State models are immutable by default. Use `.updating(...)` to create modified copies.

```python
from draive import State


class AppConfig(State):
    environment: str
    retries: int = 3


config = AppConfig(environment="staging")
updated = config.updating(retries=5)
```

## 3. Generate Text

`TextGeneration.generate(...)` is the simplest interface for text output.

```python
from draive import TextGeneration


result: str = await TextGeneration.generate(
    instructions="You are a concise assistant.",
    input="Write one sentence about typed APIs.",
)
```

## 4. Generate Structured Output

When your downstream code needs strong contracts, generate typed serializable state.

```python
from draive import ModelGeneration, State


class Order(State, serializable=True):
    id: str
    total: float
    currency: str


order: Order = await ModelGeneration.generate(
    Order,
    instructions="Extract order fields from the input.",
    input="Order #A-42 totals 129.90 USD.",
)
```

## 5. Register Tools

Tools are async functions decorated with `@tool`. They can be passed directly to generation calls.

```python
from datetime import datetime, timezone

from draive import TextGeneration, tool


@tool(description="Return current UTC timestamp")
async def current_time() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


answer = await TextGeneration.generate(
    instructions="Use tools when needed.",
    input="What time is it now?",
    tools=[current_time],
)
```

## 6. Add Retrieval With `VectorIndex`

`VectorIndex` is a context state API. In this example we use the in-memory implementation from
`VolatileVectorIndex()`.

```python
from collections.abc import Sequence

from draive import State, VectorIndex, ctx
from draive.helpers import VolatileVectorIndex


class Chunk(State, serializable=True):
    text: str


chunks: Sequence[Chunk] = (
    Chunk(text="Draive uses scoped state."),
    Chunk(text="Vector indexes support semantic search."),
)


async with ctx.scope("retrieval", VolatileVectorIndex()):
    await VectorIndex.index(Chunk, values=chunks, attribute=Chunk._.text)
    hits: Sequence[Chunk] = await VectorIndex.search(Chunk, query="semantic", limit=2)
```

## 7. Run Moderation Guardrails

If your active provider registers moderation state, you can run guardrail checks directly.

```python
from draive.guardrails import GuardrailsModeration
from draive.multimodal import MultimodalContent

await GuardrailsModeration.check_input(MultimodalContent.of("some user content"))
```

## Next Steps

1. Continue with [Basic Usage](../guides/BasicUsage.md).
1. Explore [Basic RAG](../cookbooks/BasicRAG.md).
1. Add quality checks with [Basic Evaluation](../guides/BasicEvaluation.md).
