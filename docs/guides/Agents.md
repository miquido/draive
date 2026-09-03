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
- `AgentThread` is the scoped runtime state propagated through `ctx.scope(...)`: its `identifier`
    names the conversation thread shared across nested agent calls, while `agent_uri` marks the
    URI of the agent currently executing within it - agents bind a thread stamped with their own
    URI while handling a message.
- `AgentExecuting` is the executor protocol:
    `AgentMessage -> AsyncGenerator[MultimodalContentPart | ProcessingEvent]`.

In other words, `Agent` itself is not a stateful conversation object. It is an immutable wrapper
that runs an executor inside a scoped agent context and streams output.

## 1. Build An Agent From `Step`s

Use `Agent.steps(...)` when you already have a `Step` pipeline and want to expose it as an
agent.

```python
from collections.abc import AsyncGenerator

from draive import Agent, ProcessingEvent
from draive.multimodal import TextContent
from draive.steps import Step, StepState


async def execute(
    state: StepState,
) -> AsyncGenerator[ProcessingEvent | TextContent | StepState]:
    yield ProcessingEvent.of("progress", "Analyzing request...")
    yield TextContent.of("Done")
    yield state


worker: Agent = Agent.steps(
    Step(execute),
    agent="worker",
    description="Handles a small processing task",
)
```

Call the agent inside a context scope and consume the stream.

```python
from collections.abc import AsyncGenerator

from draive import ctx
from draive.multimodal import MultimodalContentPart
from draive.utils import ProcessingEvent


async with ctx.scope("agents.step"):
    stream: AsyncGenerator[MultimodalContentPart | ProcessingEvent] = worker.call(
        input="Please help"
    )
    async for chunk in stream:
        print(chunk)
```

If you need lower-level control, build `AgentMessage` yourself and call `respond(...)` directly.

```python
from collections.abc import AsyncGenerator

from draive import AgentMessage
from draive.multimodal import MultimodalContentPart
from draive.utils import ProcessingEvent


message: AgentMessage = AgentMessage.of("Please help")

async with ctx.scope("agents.respond"):
    stream: AsyncGenerator[MultimodalContentPart | ProcessingEvent] = worker.respond(
        message
    )
    async for chunk in stream:
        print(chunk)
```

### What `steps(...)` Does

- Seeds the initial pipeline context with the incoming agent message as a single `ModelInput`,
    carrying over the message metadata.
- Executes your steps as one `Step.sequence(...)` and streams it.
- Filters out `ModelReasoningChunk`.
- Filters out `ModelToolRequest` and `ModelToolResponse` chunks.
- Streams only user-visible content and `ProcessingEvent`s.

This makes step-backed agents a good fit when you want deterministic orchestration and typed state
updates, but a clean public output stream.

One important implication: if your step emits reasoning or tool protocol chunks, callers of the
agent will not see them - those chunks are dropped from the public stream. `Agent.steps(...)` is
intentionally a public-facing wrapper over a more verbose internal step stream.

## 2. Build A Generative Model-Backed Agent

Use `Agent.generative(...)` when the agent should directly call the configured
`GenerativeModel.completion(...)`.

```python
from collections.abc import AsyncGenerator

from draive import Agent, ctx, load_env, tool
from draive.multimodal import MultimodalContentPart
from draive.openai import OpenAI, OpenAIResponsesConfig
from draive.utils import ProcessingEvent


load_env()


@tool(description="Return current system status")
async def system_status() -> str:
    return "All systems operational"


assistant: Agent = Agent.generative(
    agent="support",
    description="Answers product support questions",
    instructions="You are a concise support assistant. Use tools when useful.",
    tools=[system_status],
)


async with ctx.scope(
    "agents.generative",
    OpenAIResponsesConfig(model="gpt-5.5"),
    disposables=(OpenAI(),),
):
    stream: AsyncGenerator[MultimodalContentPart | ProcessingEvent] = assistant.call(
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

By default the model context is local to a single request; the agent does not persist prior turns
by itself. Pass `memory=` to keep context across turns (see below), use higher level conversation
APIs, or provide the required context explicitly.

### Persist Context Across Turns With `AgentMemory`

`Agent.generative(...)` and `Agent.from_skill(...)` accept a `memory` argument controlling how
model context is prepared and recalled before each turn and persisted afterwards. The default is
`AgentMemory.disabled`, which scopes context to a single turn. `Agent.steps(...)` applies no memory
implicitly - compose the predefined memory steps (`memory.recall_step()`,
`memory.remember_step()`) into the pipeline where needed; `Agent.generative(...)` and
`Agent.from_skill(...)` invoke `memory.prepare`, `memory.recall`, and `memory.remember` directly
within their step bodies instead. There is no `prepare` step - a step cannot contribute the tools
it produces to the toolbox of a subsequent completion step - so preparation runs only in
`Agent.generative(...)` and `Agent.from_skill(...)`.

```python
from draive import Agent, AgentMemory

assistant: Agent = Agent.generative(
    agent="support",
    instructions="You are a concise support assistant.",
    memory=AgentMemory.volatile(threads_limit=32),
)
```

With `Agent.steps(...)` the same memory is composed explicitly - recall before the work, remember
after it.

```python
from draive import Agent, AgentMemory, ModelOutput
from draive.multimodal import MultimodalContent
from draive.steps import StepState, step


memory: AgentMemory = AgentMemory.volatile()


@step
async def reply(
    state: StepState,
) -> StepState:
    return state.appending_context(ModelOutput.of(MultimodalContent.of("reply")))


worker: Agent = Agent.steps(
    memory.recall_step(),
    reply,
    memory.remember_step(),
    agent="worker",
)
```

`recall_step()` replaces `StepState.context` with the recalled context, `remember_step()` persists
the current `StepState.context` and leaves state unchanged. Running recall twice within one
pipeline duplicates stored history; that is a composition error.

Memory operations receive the active `AgentThread` - the executing agent's URI together with the
conversation thread identifier - so one memory instance can serve multiple agents and multiple
concurrent conversation threads, as long as the implementation keys stored context by both (the
built-in ones do). Context is stored as the latest snapshot per agent and thread: whatever is
remembered after a turn becomes the next recalled context, exactly as provided, which allows steps
to compact, summarize, or replace the context freely.

- `AgentMemory.volatile(...)` keeps snapshots in-process, optionally seeded with an `initial`
    context for new agent-thread entries, and with optional LRU-like eviction via `threads_limit`.
    An entry's usage is refreshed by `prepare` and `remember`, but not by `recall`; intended for
    local development, tests, and single-process deployments.
- `PostgresAgentMemory.instance()` (from `draive.postgres`) persists immutable snapshots in
    PostgreSQL, keyed by executing agent URI and thread; run `PostgresAgentMemory.migrate()` once
    to create its schema.
- `AgentMemory(recalling=..., remembering=...)` wraps custom async callables for any other
    backend; each receives the executing `AgentThread`, and an optional `preparing=` callable runs
    before each recall to set up state based on the agent's instructions (it must be idempotent
    per agent and thread).

Three behavioral notes:

- Context is remembered only when a turn completes and its output stream is fully consumed;
    turns abandoned mid-stream or failing with an error are not persisted.
- Memory steps (`recall_step`, `remember_step`) require an `AgentThread` bound in the current
    context and raise `AgentException` otherwise. Agents bind one stamped with their own URI
    automatically; when running memory steps outside an agent, enter a scope with an `AgentThread`
    instance first.
- `preparing=` may return tools (`Sequence[Tool] | None`); see below.

#### Let Memory Contribute Tools

`preparing=` can return tools to extend the agent toolbox for the prepared turn, which lets memory
expose its own operations to the model - looking up older context, recording a fact - without the
agent knowing about the backend. Returned tools may close over whatever preparation resolved, so no
extra keying by agent and thread is needed inside them.

```python
from collections.abc import Sequence
from typing import Any

from draive import AgentThread, ModelInstructions, Tool, tool


async def preparing(
    thread: AgentThread,
    instructions: ModelInstructions,
    **extra: Any,
) -> Sequence[Tool]:
    @tool(name="memory_search")
    async def memory_search(query: str) -> str:
        # the tool closes over the prepared turn scope
        return await search_archive(
            query,
            agent_uri=thread.agent_uri,
            thread=thread.identifier,
        )

    return (memory_search,)


# passed as AgentMemory(recalling=..., remembering=..., preparing=preparing)
```

- Prepared tools live for one turn and stay available across all of its tool-calling iterations.
- They are merged into the toolbox passed to the agent and **replace** provided tools using the
    same names, so prefix memory tool names to avoid shadowing agent tools by accident.
- They participate in the toolbox suggestion strategy; with `Toolbox.of(..., suggesting=True)` the
    model may be required to call one of them.
- Returning `None` or an empty sequence contributes nothing and leaves the toolbox untouched.
- Requests and responses of prepared tools are part of the context passed to `remember`, so a
    stored turn may reference a tool that a later turn no longer offers. Keep tool names stable, or
    drop those context elements before remembering.

## 3. Preserve Thread And Metadata

`Agent.call(...)` automatically reuses the current `AgentThread` when present. That allows nested
agent calls to share a logical thread and metadata.

```python
from collections.abc import AsyncGenerator
from uuid import uuid4

from draive import Agent, AgentIdentity, AgentMessage, ctx
from draive.agents import AgentThread
from draive.multimodal import MultimodalContentPart, TextContent
from draive.utils import ProcessingEvent


async def echo(
    message: AgentMessage,
) -> AsyncGenerator[MultimodalContentPart | ProcessingEvent]:
    context = ctx.state(AgentThread)
    yield TextContent.of(
        f"thread={context.identifier} source={context.meta.get_str('source')}"
    )


agent: Agent = Agent(
    identity=AgentIdentity.of(name="echo"),
    executing=echo,
)


async with ctx.scope(
    "agents.context",
    AgentThread.of(identifier=uuid4(), agent_uri="agent://outer", meta={"source": "outer"}),
):
    stream: AsyncGenerator[MultimodalContentPart | ProcessingEvent] = agent.call(
        input="hello",
        meta={"request": "nested"},
    )
    async for chunk in stream:
        print(chunk)
```

In practice:

- `thread=` on `call(...)` overrides the current context thread,
- `meta=` is merged with the current `AgentThread.meta`,
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
    agent="researcher",
    description="Collects background information",
)

writer: Agent = Agent.generative(
    agent="writer",
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
    agent="researcher",
    description="Finds facts and prepares structured findings",
    instructions="Gather only the information needed for the task.",
)

reviewer: Agent = Agent.generative(
    agent="reviewer",
    description="Checks output for completeness and correctness",
    instructions="Review the provided answer and suggest corrections.",
)

coordinator: Agent = Agent.generative(
    agent="coordinator",
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
- Use `Agent.from_skill(...)` to turn a loaded `Skill` into a model-backed agent with access to its
    bundled resources - see [Skills](./Skills.md).
- Expose one concrete agent via `Agent.as_tool(...)` when a model should call it directly.
- Delegate using `AgentsGroup.as_tool(handling="response")` when the caller should continue reasoning after delegation.
- Delegate using `AgentsGroup.as_tool(handling="output")` when the delegated agent should take over visible output.

Avoid using `Agent.generative(...)` as a substitute for persistent chat history. By design it loops
only within one request while tools are being resolved; use `memory=` when prior turns should be
recalled across requests.

## 8. Public Types

The public agents API exported from `draive` includes:

- `Agent`
- `AgentException`
- `AgentExecuting`
- `AgentIdentity`
- `AgentMemory`
- `AgentMessage`
- `AgentThread`
- `AgentUnavailable`
- `AgentsGroup`
- `ProcessingEvent`

The memory protocols are exported from `draive.agents` only:

- `AgentMemoryPreparing`
- `AgentMemoryRecalling`
- `AgentMemoryRemembering`
