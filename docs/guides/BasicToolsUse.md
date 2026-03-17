# Basic Tools Use

Tools are async Python callables wrapped with `@tool` and exposed to models through
`TextGeneration` or `ModelGeneration`.

## 1. Define A Tool

```python
from draive import tool


@tool(description="Return local time for a city")
async def current_time(location: str) -> str:
    # In real usage call an external API or internal service.
    return f"Time in {location} is 09:53:22"
```

You can call a tool directly, but only inside a context scope.

```python
from draive import ctx


async with ctx.scope("tools"):
    print(await current_time(location="London"))
```

## 2. Expose Tools To Generation

When `tools=[...]` is passed, Draive handles request/response tool turns automatically.

```python
from draive import TextGeneration, ctx, load_env
from draive.openai import OpenAI, OpenAIResponsesConfig

load_env()

async with ctx.scope(
    "tools",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    result: str = await TextGeneration.generate(
        instructions="You are a helpful assistant.",
        input="What is the time in New York?",
        tools=[current_time],
    )
    print(result)
```

## 3. Customize Metadata And Argument Schema

`Alias` and `Description` annotations shape the model-facing tool specification.

```python
from typing import Annotated

from draive import Alias, Description, tool


@tool(name="fun_fact", description="Find a fun fact for a topic")
async def customized(
    topic: Annotated[str, Alias("topic"), Description("Topic of a fact to find")],
) -> str:
    return f"{topic} is fun."


# Useful for debugging what model sees.
print(customized.specification)
```

## 4. Use `Toolbox` For Selection Strategy

`Toolbox` lets you compose tools and provide suggestion strategy for the first model turn.

```python
from draive import TextGeneration, Toolbox


toolbox = Toolbox.of([current_time, customized], suggesting=customized)

result = await TextGeneration.generate(
    instructions="Be helpful and use tools when needed.",
    input="Share one fun fact about LLMs.",
    tools=toolbox,
)
```

## 5. Advanced Tool Behavior

`@tool(...)` supports:

- `availability=...` for runtime availability checks
- `result_formatting=...` to transform success output
- `error_formatting=...` to shape tool failure content
- `handling="response" | "output" | "detached"` to control model orchestration behavior
