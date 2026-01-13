# Quickstart

Build a minimal Draive app using `ctx.scope`, provider configuration, tool calls, and typed output.

## Prerequisites

```bash
pip install "draive[openai]"
```

Create `.env`:

```env
OPENAI_API_KEY=your-api-key
```

Load env variables before your first provider call:

```python
from draive import load_env

load_env()
```

## Generate Text

This is the smallest production-shaped flow:

- scope label for logs/traces,
- explicit model configuration state,
- provider client managed as disposable.

```python
import asyncio

from draive import TextGeneration, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


async def main() -> None:
    async with ctx.scope(
        "quickstart",
        OpenAIResponsesConfig(model="gpt-5-mini"),
        disposables=(OpenAI(),),
    ):
        result: str = await TextGeneration.generate(
            instructions="You are a helpful assistant",
            input="What is the capital of France?",
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

## Add Tools

Tools are regular async functions wrapped with `@tool`. Pass them through `tools=[...]` and the
framework will manage request/response tool turns with the model.

```python
from datetime import datetime, timezone

from draive import TextGeneration, ctx, tool
from draive.openai import OpenAI, OpenAIResponsesConfig


@tool(description="Return current UTC timestamp")
async def current_time() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


async def with_tools() -> None:
    async with ctx.scope(
        "assistant",
        OpenAIResponsesConfig(model="gpt-5-mini"),
        disposables=(OpenAI(),),
    ):
        result = await TextGeneration.generate(
            instructions="Use tools when useful.",
            input="What time is it right now?",
            tools=[current_time],
        )
        print(result)
```

## Generate Typed Output

Use `ModelGeneration` when you want structured output that is validated and decoded into a typed
model.

```python
from collections.abc import Sequence

from draive import ModelGeneration, State, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


class PersonInfo(State, serializable=True):
    name: str
    age: int
    occupation: str
    skills: Sequence[str]


async def extract_person() -> None:
    async with ctx.scope(
        "extraction",
        OpenAIResponsesConfig(model="gpt-5-mini"),
        disposables=(OpenAI(),),
    ):
        person: PersonInfo = await ModelGeneration.generate(
            PersonInfo,
            instructions="Extract person details from the passage.",
            input=(
                "John Smith is a 32-year-old software engineer in Seattle. "
                "He works with Python, ML, and cloud architecture."
            ),
        )
        print(person)
```

## Swap Providers

The surrounding workflow stays the same; only config/client states change.

```python
from draive import TextGeneration, ctx
from draive.anthropic import Anthropic, AnthropicConfig
from draive.gemini import Gemini, GeminiConfig


async def joke_with_claude() -> None:
    async with ctx.scope(
        "claude",
        AnthropicConfig(model="claude-3-5-haiku-20241022"),
        disposables=(Anthropic(),),
    ):
        print(
            await TextGeneration.generate(
                instructions="You are concise and playful.",
                input="Tell me a short joke.",
            )
        )


async def joke_with_gemini() -> None:
    async with ctx.scope(
        "gemini",
        GeminiConfig(model="gemini-2.5-flash"),
        disposables=(Gemini(),),
    ):
        print(
            await TextGeneration.generate(
                instructions="You are concise and playful.",
                input="Tell me a short joke.",
            )
        )
```

## Next Steps

- [First Steps](./first-steps.md)
- [Basic Model Generation](../guides/BasicModelGeneration.md)
- [Basic Tools Use](../guides/BasicToolsUse.md)
