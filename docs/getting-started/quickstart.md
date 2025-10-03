# Quickstart

Build a minimal Draive app that talks to OpenAI, loads environment variables, invokes tools, and
adapts to other providers.

## Prerequisites

```bash
pip install "draive[openai]"
```

Create an `.env` file with your credentials:

```env
OPENAI_API_KEY=your-api-key-here
```

Load the environment before hitting the API:

```python
from draive import load_env

load_env()
```

## Generate your first response

```python
import asyncio
from draive import ctx, TextGeneration
from draive.openai import OpenAI, OpenAIResponsesConfig

async def main() -> None:
    async with ctx.scope(
        "quickstart",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        result = await TextGeneration.generate(
            instructions="You are a helpful assistant",
            input="What is the capital of France?",
        )
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Expose tools to the model

```python
from datetime import datetime, timezone
from draive import ctx, TextGeneration, tool
from draive.openai import OpenAI, OpenAIResponsesConfig

@tool(description="Return the current UTC timestamp")
async def current_time() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

async def with_tools() -> None:
    async with ctx.scope(
        "assistant",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        result = await TextGeneration.generate(
            instructions="You are a helpful assistant with access to tools",
            input="What's the time right now?",
            tools=[current_time],
        )
        print(result)
```

## Generate structured data

```python
from collections.abc import Sequence
from draive import DataModel, ModelGeneration, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig

class PersonInfo(DataModel):
    name: str
    age: int
    occupation: str
    skills: Sequence[str]

async def extract_person() -> None:
    async with ctx.scope(
        "extraction",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        person = await ModelGeneration.generate(
            PersonInfo,
            instructions="Extract the person details from the passage",
            input="""
            John Smith is a 32-year-old software engineer living in Seattle.
            He specialises in Python, machine learning, and cloud architecture.
            """,
        )
        print(person)
```

## Swap providers

```python
from draive.anthropic import Anthropic, AnthropicConfig
from draive.gemini import Gemini, GeminiConfig
from draive import TextGeneration, ctx

async def joke_with_claude() -> None:
    async with ctx.scope(
        "claude",
        AnthropicConfig(model="claude-3-5-haiku-20241022"),
        disposables=(Anthropic(),),
    ):
        result = await TextGeneration.generate(
            instructions="You are Claude, a playful assistant",
            input="Tell me a short joke",
        )
        print(result)

async def joke_with_gemini() -> None:
    async with ctx.scope(
        "gemini",
        GeminiConfig(model="gemini-2.5-flash"),
        disposables=(Gemini(),),
    ):
        result = await TextGeneration.generate(
            instructions="You are Gemini, a playful assistant",
            input="Tell me a short joke",
        )
        print(result)
```

## Next steps

- Wire up conversations with memory in [Basic Conversation](../guides/BasicConversation.md).
- Explore typed outputs in [Basic Model Generation](../guides/BasicModelGeneration.md).
- Manage observability and traces with [Evaluator Catalog](../guides/EvaluatorCatalog.md).

When you are ready for deeper patterns continue with the [First Steps](first-steps.md) guide.
