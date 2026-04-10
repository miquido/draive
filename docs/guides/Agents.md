# Agents

`Agent` is Draive's lightweight abstraction for building reusable async workers that:

- accept typed multimodal input,
- stream visible output chunks and `ProcessingEvent`s,
- preserve conversation thread and metadata through `ctx.scope(...)`, and
- can be exposed to other agents as tools through `AgentsGroup`.

The package is intentionally small. It builds on top of existing Draive primitives:

- `Step` for custom execution pipelines,
- `GenerativeModel` for model-backed execution,
- `Toolbox` for tool handling,
- `MultimodalContent` for input and output transport.

## Runtime Model

The agents API is intentionally built as a thin layer over existing Draive runtime abstractions.

- `AgentIdentity` describes the agent instance: `uri`, `name`, `description`, `meta`.
- `AgentMessage` is the fully prepared input payload: `thread`, `created`, `content`, `meta`.
- `AgentContext` is the scoped runtime state propagated through `ctx.scope(...)`.
- `AgentExecuting` is the executor protocol:
    `AgentMessage -> AsyncIterable[MultimodalContentPart | ProcessingEvent]`.

In other words, `Agent` itself is not a stateful conversation object. It is an immutable wrapper
that runs an executor inside a scoped agent context and streams output.

## 1. Build An Agent From `Step`s

Use `Agent.steps(...)` when you already have a `Step` pipeline and want to expose it as an
agent.

```python
from collections.abc import AsyncIterable

from draive import Agent, ProcessingEvent
from draive.multimodal import TextContent
from draive.steps import Step, StepState


async def execute(
    state: StepState,
) -> AsyncIterable[ProcessingEvent | TextContent | StepState]:
    yield ProcessingEvent.of("progress", "Analyzing request...")
    yield TextContent.of("Done")
    yield state


worker: Agent = Agent.steps(
    Step(execute),
    name="worker",
    description="Handles a small processing task",
)
```

Call the agent inside a context scope and consume the stream.

```python
from collections.abc import AsyncIterable

from draive import ctx
from draive.multimodal import MultimodalContentPart
from draive.utils import ProcessingEvent


async with ctx.scope("agents.step"):
    stream: AsyncIterable[MultimodalContentPart | ProcessingEvent] = worker.call(
        input="Please help"
    )
    async for chunk in stream:
        print(chunk)
```

If you need lower-level control, build `AgentMessage` yourself and call `respond(...)` directly.

```python
from collections.abc import AsyncIterable

from draive import AgentMessage
from draive.multimodal import MultimodalContentPart
from draive.utils import ProcessingEvent


message: AgentMessage = AgentMessage.of("Please help")

async with ctx.scope("agents.respond"):
    stream: AsyncIterable[MultimodalContentPart | ProcessingEvent] = worker.respond(
        message
    )
    async for chunk in stream:
        print(chunk)
```

### What `steps(...)` Does

- Prepends the incoming agent message into step state with `Step.appending_input(...)`.
- Executes your step pipeline.
- Filters out `ModelReasoningChunk`.
- Treats leaked `ModelToolRequest` and `ModelToolResponse` chunks as an internal contract violation.
- Streams only user-visible content and `ProcessingEvent`s.

This makes step-backed agents a good fit when you want deterministic orchestration and typed state
updates, but a clean public output stream.

One important implication: if your step emits reasoning, callers of the agent will not see it. If
your step emits tool protocol chunks, `Agent.steps(...)` will raise `AssertionError` in debug mode
instead of exposing them to callers. `Agent.steps(...)` is intentionally a public-facing wrapper
over a more verbose internal step stream.

## 2. Build A Generative Model-Backed Agent

Use `Agent.generative(...)` when the agent should directly call the configured
`GenerativeModel.completion(...)`.

```python
from collections.abc import AsyncIterable

from draive import Agent, ctx, load_env, tool
from draive.multimodal import MultimodalContentPart
from draive.openai import OpenAI, OpenAIResponsesConfig
from draive.utils import ProcessingEvent


load_env()


@tool(description="Return current system status")
async def system_status() -> str:
    return "All systems operational"


assistant: Agent = Agent.generative(
    name="support",
    description="Answers product support questions",
    instructions="You are a concise support assistant. Use tools when useful.",
    tools=[system_status],
)


async with ctx.scope(
    "agents.generative",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    stream: AsyncIterable[MultimodalContentPart | ProcessingEvent] = assistant.call(
        input="Check the current system status"
    )
    async for chunk in stream:
        print(chunk)
```

### How The Generative Loop Works

For each call, the agent:

1. converts the incoming message into `ModelInput`,
1. calls `GenerativeModel.completion(...)`,
1. collects any `ModelToolRequest`s,
1. executes them through `Toolbox.handle(...)`,
1. appends `ModelToolResponse`s back into model context,
1. repeats until the model produces a final answer.

If a tool uses `handling="output"`, the tool can stream visible output directly and terminate the
loop early.

This API is request-scoped because the model context is local to a single request. The agent
does not persist prior turns by itself. If you need longer-lived conversation semantics, use higher
level conversation APIs or pass the required context explicitly.

## 3. Preserve Thread And Metadata

`Agent.call(...)` automatically reuses the current `AgentContext` when present. That allows nested
agent calls to share a logical thread and metadata.

```python
from collections.abc import AsyncIterable
from uuid import uuid4

from draive import Agent, AgentIdentity, AgentMessage, ctx
from draive.agents.types import AgentContext
from draive.multimodal import MultimodalContentPart, TextContent
from draive.utils import ProcessingEvent


async def echo(
    message: AgentMessage,
) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
    context = ctx.state(AgentContext)
    yield TextContent.of(
        f"thread={context.thread} source={context.meta.get_str('source')}"
    )


agent: Agent = Agent(
    identity=AgentIdentity.of(name="echo"),
    executing=echo,
)


async with ctx.scope(
    "agents.context",
    AgentContext.of(thread=uuid4(), meta={"source": "outer"}),
):
    stream: AsyncIterable[MultimodalContentPart | ProcessingEvent] = agent.call(
        input="hello",
        meta={"request": "nested"},
    )
    async for chunk in stream:
        print(chunk)
```

In practice:

- `thread=` on `call(...)` overrides the current context thread,
- `meta=` is merged with the current `AgentContext.meta`,
- `respond(...)` is useful when you already have a prepared `AgentMessage`.

This matters when agents call other agents. Nested calls inherit the active thread and metadata by
default, which makes it easier to correlate delegation chains in observability and preserve request
context without global state.

## 4. Expose Agents As Tools

`AgentsGroup` lets one agent delegate work to another using the same tool infrastructure used for
regular model tools.

```python
from draive import Agent, AgentsGroup
from draive.multimodal import TextContent
from draive.steps import Step


researcher: Agent = Agent.steps(
    Step.emitting(TextContent.of("Collected facts")),
    name="researcher",
    description="Collects background information",
)

writer: Agent = Agent.generative(
    name="writer",
    description="Writes the final response",
    instructions="Delegate research first, then answer clearly.",
    tools=[AgentsGroup.of(researcher).as_tool()],
)
```

The generated tool takes:

- `agent`: selected agent name,
- `task`: plain-text request sent to that agent.

Agent names must be unique inside a group. `AgentsGroup.of(...)` raises `ValueError` if duplicate
names are provided.

You can also expose a single agent directly as a tool when you do not need a group registry.

```python
from draive.tools import Tool


tools: list[Tool] = [
    researcher.as_tool(),
]
```

## 5. Choose Response vs Output Handling

Both `Agent.as_tool(...)` and `AgentsGroup.as_tool(...)` support two delegation modes through the
`handling=` argument.

### `handling="response"`

Use `handling="response"` when the delegated agent result should come back as a normal tool
response and be fed into the caller's model loop.

```python
from draive.tools import Tool


tools: Tool = AgentsGroup.of(researcher).as_tool(handling="response")
```

This behaves like any regular `handling="response"` tool.

Choose this mode when the caller agent should inspect the delegated result and continue its own
reasoning loop.

### `handling="output"`

Use `handling="output"` when the delegated agent should stream final output directly to the user.

```python
from draive.tools import Tool


tools: Tool = AgentsGroup.of(researcher).as_tool(handling="output")
```

This behaves like a `handling="output"` tool. Output chunks from the selected agent are streamed
immediately, and the tool still finishes by returning a `ModelToolResponse` with
`handling="output"`.

Choose this mode when delegation should feel like a transfer of control rather than a background
lookup.

## 6. Typical Multi-Agent Pattern

```python
from draive import Agent, AgentsGroup


researcher: Agent = Agent.generative(
    name="researcher",
    description="Finds facts and prepares structured findings",
    instructions="Gather only the information needed for the task.",
)

reviewer: Agent = Agent.generative(
    name="reviewer",
    description="Checks output for completeness and correctness",
    instructions="Review the provided answer and suggest corrections.",
)

coordinator: Agent = Agent.generative(
    name="coordinator",
    description="Routes tasks between specialized agents",
    instructions=(
        "Use `agent_request` to delegate work to the specialized agents. "
        "Combine their results into one final answer."
    ),
    tools=[AgentsGroup.of(researcher, reviewer).as_tool()],
)
```

This pattern works well when:

- one model-facing agent orchestrates the workflow,
- specialized agents have narrow responsibilities,
- you want delegation without introducing a separate orchestration framework.

Keep the specialized agents narrow. `AgentsGroup` is most useful when each delegated agent has a
clear role and the coordinator can select among them by name from the generated tool schema.

## 7. When To Use Each API

- Prefer `Agent.steps(...)` for deterministic pipelines, typed artifacts, and explicit control.
- Try `Agent.generative(...)` for prompt-first, tool-aware model agents.
- Expose one concrete agent via `Agent.as_tool(...)` when a model should call it directly.
- Delegate using `AgentsGroup.as_tool(handling="response")` when the caller should continue reasoning after delegation.
- Delegate using `AgentsGroup.as_tool(handling="output")` when the delegated agent should take over visible output.

Avoid using `Agent.generative(...)` as a substitute for persistent chat history. By design it loops
only within one request while tools are being resolved.

## 8. Public Types

The public agents API exported from `draive` includes:

- `Agent`
- `AgentExecuting`
- `AgentIdentity`
- `AgentMessage`
- `ProcessingEvent`
- `AgentsGroup`

Additional agent-specific exceptions are available from `draive.agents`:

- `AgentException`
- `AgentUnavailable`
