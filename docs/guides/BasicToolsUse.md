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

You can call a tool directly like a regular async function. A context scope is required only
when the tool itself uses context state, logging or observability.

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
    OpenAIResponsesConfig(model="gpt-5.5"),
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

- `name=...` and `description=...` to override the model facing tool identity
- `parameters=...` to provide an explicit arguments JSON schema instead of the inferred one
- `handling="response" | "output" | "output_stream"` to control whether the tool result only goes
    back to the model or is also surfaced in the output stream
- `meta=...` to attach metadata to the tool specification
- `meta={"strict_parameters": True}` to have the provider enforce the argument schema, which
    makes the model always pass every argument rather than relying on its default. A tool taking
    a mapping argument keeps its arguments unenforced, since enforcement would drop the mapping

A tool function can also be an async generator yielding content parts and `ProcessingEvent` values,
which allows streaming partial results while the tool is still running. Streaming tools have to be
async generators - not arbitrary async iterables - so that a `try/finally` can span the whole stream
and release scoped state deterministically when a consumer stops early. `Toolbox.handle(...)` closes
the tool stream it consumed; when consuming `tool.call(...)` directly and stopping before the end,
close it explicitly instead of leaving it to garbage collection.
