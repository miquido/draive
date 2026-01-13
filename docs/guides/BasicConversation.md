# Basic Conversation

Use `Conversation.completion(...)` for chat-style flows with optional tool support.

Compared with one-shot text generation, conversation mode is better when you want streaming chunks,
tool events, and conversation-oriented orchestration.

## Prerequisites

- `pip install "draive[openai]"`
- `.env` with `OPENAI_API_KEY`

```python
from draive import load_env

load_env()
```

## Define An Optional Tool

```python
from datetime import UTC, datetime

from draive import tool


@tool(description="Return UTC date and time")
async def utc_datetime() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
```

## Run Conversation Streaming

`Conversation.completion(...)` returns an async stream. Consume it chunk-by-chunk for UI/realtime
usage.

```python
from draive import Conversation, ctx
from draive.tools import Toolbox
from draive.openai import OpenAI, OpenAIResponsesConfig


async with ctx.scope(
    "conversation",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    stream = Conversation.completion(
        instructions="You are a helpful assistant.",
        message="Hi! What time is it now?",
        toolbox=Toolbox.of([utc_datetime]),
    )

    async for chunk in stream:
        # Chunk type may be content, reasoning, or tool event.
        print(chunk)
```

## When To Use This API

Use `Conversation` when you need:

- incremental output consumption,
- explicit tool request/response events,
- conversation-specific orchestration semantics.

Use `TextGeneration.generate(...)` for simpler one-shot text output.
