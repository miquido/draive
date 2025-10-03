# Basic Conversation with Draive and OpenAI

This guide walks you through a minimal yet complete conversation flow using Draive's conversation
helpers and OpenAI as the model provider. By the end you will know how to load secrets, register
simple tools, and request a response from the model inside a managed Haiway context.

## Prerequisites

- Python 3.12 or newer.
- Project dependencies installed (for example `uv sync`).
- An `.env` file with `OPENAI_API_KEY` set so Draive can authenticate against OpenAI.

> 💡 Draive never reads environment variables directly. Always load them through `draive.load_env()`
> so secrets are available to Haiway states.

## 1. Load environment secrets

Use `draive.load_env()` at startup to populate the process environment from the local `.env` file.

```python
from draive import load_env

load_env()  # pulls OPENAI_API_KEY (and other variables) into the session
```

## 2. Define optional tools

Tools extend the model with deterministic abilities such as retrieving the current time. Define them
with the `@tool` decorator and standard Python code. Tools must be async functions.

```python
from datetime import UTC, datetime

from draive import tool

@tool(description="UTC time and date now")
async def utc_datetime() -> str:
    """Return the current date and time formatted as a readable string."""
    return datetime.now(UTC).strftime("%A %d %B, %Y, %H:%M:%S")
```

## 3. Run a conversation

Inside a `ctx.scope(...)` block you can compose the state required for the conversation. Provide
configuration for the OpenAI model, instantiate the OpenAI client as a disposable resource, and
finally call `Conversation.completion(...)` with your prompt and optional tools.

```python
from draive import Conversation, ConversationMessage, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig

async with ctx.scope(
    "basics",  # scope name visible in logs and traces
    OpenAIResponsesConfig(model="gpt-3.5-turbo-0125"),  # model configuration
    disposables=(OpenAI(),),  # lifecycle-managed OpenAI client
):
    response: ConversationMessage = await Conversation.completion(
        instructions="You are a helpful assistant.",
        input="Hi! What is the time now?",
        tools=[utc_datetime],
    )

    print(response)
```

Example output (truncated for clarity):

```
identifier: dd2a86730a3441939359e960f0cc2da3
role: model
created: 2025-03-07 12:40:30.130777+00:00
content:
  parts:
    - text: The current UTC time and date is Friday, 7th March 2025, 12:40:29.
```

The `ConversationMessage` object contains structured parts that you can inspect or render. If you
prefer to display only the assistant text, use `response.text`.

## Next steps

- Swap `OpenAIResponsesConfig` for another provider module (for example `draive.mistral`) to try
  different models.
- Add more tools to give the model controlled access to proprietary data or services.
- Wrap the code in an async function and trigger it from your application entrypoint or a CLI
  script.
