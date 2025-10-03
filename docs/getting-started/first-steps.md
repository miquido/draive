# First Steps

Build confidence with Draive by learning how scoped contexts, state containers, and typed models fit
together. This guide walks through the essential patterns you will use in every project.

## Activate your environment

```python
from draive import ctx
from draive.openai import OpenAI, OpenAIResponsesConfig

async def main() -> None:
    async with ctx.scope(
        "quickstart",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        ...
```

`ctx.scope` binds configuration and disposables to a structured context. Everything inside the scope
can access the active configuration via `ctx.state`.

## Define immutable state

```python
from haiway import State

class AppConfig(State):
    environment: str
    max_retries: int = 3
```

State instances are immutable—use `.updated()` to create tweaked copies.

```python
config = AppConfig(environment="staging")
updated = config.updated(max_retries=5)
```

## Model your data

Use `DataModel` when you need serializable data with validation and JSON Schema support.

```python
from draive import DataModel

class Order(DataModel):
    id: str
    total: float
    currency: str
```

`Order.json_schema()` generates machine-readable schemas for tool calls and structured generation.

## Generate text

```python
from draive import TextGeneration

async def tagline() -> str:
    async with ctx.scope(
        "tagline",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        return await TextGeneration.generate(
            instructions="You are a branding assistant",
            input="Create a one-line pitch for a travel planner",
        )
```

## Add tools

```python
from draive import tool
from datetime import datetime, timezone

@tool(description="Return the current ISO timestamp")
async def current_time() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
```

Pass tools to generations or conversations to let the model call them when needed.

## Manage resources with disposables

Use disposables for clients or services that require clean-up.

```python
from draive import ctx
from draive.openai import OpenAI

async with ctx.scope("app", disposables=(OpenAI(),)):
    ...  # Client is available and cleaned up automatically
```

## Compose multimodal content

```python
from draive.multimodal import MultimodalContent, TextContent

content = MultimodalContent.of(TextContent.of("Describe this diagram:"))
```

Pass `content` into generation APIs to mix text, images, and other artifacts.

## Wire retrieval

```python
from draive.embedding import VectorIndex
from draive.resources import ResourceContent

index = VectorIndex.with_hnsw()
await index.add(ResourceContent.text("internal-notes", "Use fallback provider after 3 retries."))
```

## Evaluate quality

```python
from draive.guardrails import ModerationState

async with ctx.scope("moderated", ModerationState.of(provider="openai")):
    ...
```

Moderation, quality, and privacy guardrails plug into the same scoping mechanics.

## Next steps

1. Explore the [Guides](../guides/BasicUsage.md) for focused scenarios.
1. Try applied blueprints in the [Cookbooks](../cookbooks/BasicRAG.md).
1. Reference the API from your editor with Draive's strict type hints.

Remember: keep code inside `ctx.scope`, favour immutable state, and add features incrementally.
