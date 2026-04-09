# Advanced Step Usage

This guide documents advanced `draive.steps` usage for:

- `Step` class methods,
- `step(...)` adapter,
- `StepState` helpers,
- related step protocols and type aliases.

All examples are async and keep strict typing.

## Core Imports Used In Examples

```python
from collections.abc import Iterable

from draive import State
from draive.models import (
    ModelContext,
    ModelInput,
)
from draive.multimodal import MultimodalContent
from draive.steps import (
    Step,
    StepException,
    StepOutputChunk,
    StepState,
    step,
)
from draive.tools import Toolbox
```

## StepState Basics

`Step` transforms immutable `StepState`. `StepState` contains:

- `context: ModelContext`
- `artifacts: Mapping[str, ArtifactContent]`

### `StepState.of(...)`

Create initial state from context and typed artifacts:

```python
class Flags(State):
    ready: bool


state = StepState.of(
    (ModelInput.of(MultimodalContent.of("Analyze this.")),),
    Flags(ready=True),
    session_flags=Flags(ready=False),
)
```

### `StepState.get(...)`

Read typed artifacts by default key (`ClassName`) or custom `key`:

```python
flags = state.get(Flags, required=True)
session_flags = state.get(Flags, key="session_flags", required=True)
missing = state.get(Flags, key="unknown", default=Flags(ready=False))
```

### `StepState.updating_artifacts(...)`

Return a new state with merged/replaced artifacts:

```python
updated = state.updating_artifacts(Flags(ready=False), extra=Flags(ready=True))
```

### `StepState.appending_context(...)`

Append context elements:

```python
extended = state.appending_context(
    ModelInput.of(MultimodalContent.of("Follow-up user input"))
)
```

### `StepState.replacing_context(...)`

Replace whole context:

```python
replaced = state.replacing_context((ModelInput.of(MultimodalContent.of("Reset")),))
```

## Building Steps

### `Step.noop`

Reusable no-op step:

```python
pipeline = Step.sequence(
    Step.noop,
)
```

### `Step.emitting(*parts)`

Emit fixed output chunks without changing state:

```python
emit_notice = Step.emitting("Working...", "\n")
```

### `Step.updating_artifacts(*artifacts, **keyed_artifacts)`

Store artifacts in state:

```python
class Metrics(State):
    retries: int


set_metrics = Step.updating_artifacts(Metrics(retries=1), run_metrics=Metrics(retries=2))
```

### `Step.appending_context(*elements)`

Append context elements:

```python
add_input = Step.appending_context(
    ModelInput.of(MultimodalContent.of("Summarize this ticket."))
)
```

### `Step.replacing_context(context)`

Replace context fully:

```python
reset_context = Step.replacing_context(
    (ModelInput.of(MultimodalContent.of("Fresh conversation context")),)
)
```

### `Step.updating_context(mutation)`

Apply async transformation to current context:

```python
async def trim_context(context: ModelContext) -> ModelContext:
    return context[-4:]


trim = Step.updating_context(trim_context)
```

### `Step.restoring_state(restoring)`

Restore a previously saved `StepState` (directly or lazily):

```python
snapshot: StepState = StepState.of()


async def restore_latest() -> StepState:
    return snapshot


restore_direct = Step.restoring_state(snapshot)
restore_lazy = Step.restoring_state(restore_latest)
```

### `Step.preserving_state(preserving)`

Persist current full state externally:

```python
saved: list[StepState] = []


async def persist_state(state: StepState) -> None:
    saved.append(state)


preserve = Step.preserving_state(persist_state)
```

### `Step.appending_input(input, *, meta=None)`

Append `ModelInput` from static value or async provider:

```python
async def read_user_input():
    return MultimodalContent.of("Please draft release notes.")


append_input = Step.appending_input(read_user_input)
append_static_input = Step.appending_input(MultimodalContent.of("Static input"))
```

### `Step.appending_output(output, *, emitting=False, meta=None)`

Append `ModelOutput` from static value or async provider; optionally emit parts:

```python
async def read_human_output():
    return MultimodalContent.of("Human-approved response.")


append_output = Step.appending_output(read_human_output, emitting=True)
append_static_output = Step.appending_output(MultimodalContent.of("Static output"))
```

### `step(processing)`

Adapter from async `StepState -> StepState` function into full `Step`:

```python
@step
async def normalize(state: StepState) -> StepState:
    return state.updating_artifacts(Metrics(retries=0))
```

## Composition Patterns

### `Step.sequence(*steps)`

Run steps in strict order:

```python
pipeline = Step.sequence(
    Step.appending_context(ModelInput.of(MultimodalContent.of("Analyze incident."))),
    Step.emitting("Running analysis..."),
    Step.updating_artifacts(Metrics(retries=0)),
)
```

### `Step.loop(*steps, condition=...)`

Repeat sequence while async condition returns `True`:

```python
async def repeat_until_ready(*, state: StepState, iteration: int) -> bool:
    flags = state.get(Flags, default=Flags(ready=False))
    return not flags.ready and iteration < 3


looped = Step.loop(
    Step.updating_artifacts(Flags(ready=True)),
    condition=repeat_until_ready,
)
```

### `Step.concurrent(*steps, merge=...)`

Run branches concurrently and merge resulting states:

```python
branch_a = Step.updating_artifacts(left=Metrics(retries=1))
branch_b = Step.updating_artifacts(right=Metrics(retries=2))


async def merge_branches(branches: Iterable[StepState]) -> StepState:
    states = tuple(branches)
    merged_context = tuple(
        element
        for branch in states
        for element in branch.context
    )
    return StepState.of(merged_context)


fan_out = Step.concurrent(branch_a, branch_b, merge=merge_branches)
```

### `Step.selection(selecting)`

Choose which step to run at runtime:

```python
async def choose_step(*, state: StepState) -> Step:
    flags = state.get(Flags, default=Flags(ready=False))
    return Step.emitting("ready") if flags.ready else Step.emitting("not-ready")


selected = Step.selection(choose_step)
```

## Model And Tool Integration

### `Step.generating_completion(...)`

Single completion call that appends generated output to context:

```python
completion_step = Step.generating_completion(
    instructions="You are a concise incident assistant.",
    tools=Toolbox.empty,
    input=MultimodalContent.of("Summarize error trends from provided logs."),
    output="auto",
)
```

### `Step.handling_tools(tools)`

Execute tool requests from the latest `ModelOutput` and append tool responses:

```python
tools_step = Step.handling_tools(Toolbox.empty)
```

### `Step.looping_completion(...)`

Loop completion + tool handling until model stops requesting tools:

```python
looping_step = Step.looping_completion(
    instructions="Call tools when needed, then provide final answer.",
    tools=Toolbox.empty,
    input=MultimodalContent.of("Diagnose failing checkout flow."),
)
```

## Execution Wrappers

Wrappers return a new step around existing behavior.

### `with_ctx(*ctx_state, disposables=())`

Inject scoped context state/resources for this step:

```python
class RequestConfig(State):
    tenant: str


wrapped = completion_step.with_ctx(RequestConfig(tenant="acme"))
```

### `with_retry(limit=1, delay=None, catching=Exception)`

Retry on selected failures:

```python
retrying = completion_step.with_retry(limit=2, delay=0.25, catching=StepException)
```

### `with_fallback(fallback, catching=Exception)`

Run fallback step when matched exception is raised:

```python
fallback = Step.emitting("Fallback response")
safe = completion_step.with_fallback(fallback, catching=StepException)
```

### `with_volatile_context()`

Discard context mutations produced by the wrapped step:

```python
volatile_context = completion_step.with_volatile_context()
```

### `with_volatile_tools()`

Discard context elements marked as containing tools:

```python
volatile_tools = looping_step.with_volatile_tools()
```

### `with_condition(condition, alternative=None)`

Run wrapped step only when condition matches:

```python
async def should_run(*, state: StepState) -> bool:
    return state.get(Flags, default=Flags(ready=False)).ready


conditional = completion_step.with_condition(
    should_run,
    alternative=Step.emitting("Skipped"),
)
```

### `with_suppressed_output()`

Keep state updates but suppress non-state emitted chunks:

```python
silent = completion_step.with_suppressed_output()
```

### `with_context_evaluation(evaluator, raise_on_failure=False)`

Evaluate `state.context` before running step:

```python
context_guard = completion_step.with_context_evaluation(
    evaluator=scenario_context_guard,  # PreparedEvaluator[ModelContext]
    raise_on_failure=True,
)
```

### `with_output_evaluation(evaluator, raise_on_failure=False)`

Evaluate emitted output incrementally while step runs:

```python
output_guard = completion_step.with_output_evaluation(
    evaluator=scenario_output_guard,  # PreparedEvaluator[Sequence[StepOutputChunk]]
    raise_on_failure=True,
)
```

## Execution APIs

### `await step.run(...)`

Collect emitted multimodal chunks as `MultimodalContent`:

```python
content = await pipeline.run()
```

### `await step.process(...)`

Return final `StepState`:

```python
final_state = await pipeline.process()
```

### `async for chunk in step.stream(...)`

Stream non-state output chunks:

```python
async for chunk in pipeline.stream():
    print(chunk)
```

### `async for chunk in step`

`Step` is async-iterable and proxies `stream()`:

```python
async for chunk in pipeline:
    print(chunk)
```

## Related Types And Protocols

These contracts define typed extension points:

- `StepOutputChunk`: union of multimodal parts, reasoning chunks, tool request/response, and processing events.
- `StepStream`: async iterable yielding `StepOutputChunk | StepState`.
- `StepExecuting`: low-level callable protocol used by `Step`.
- `StepProcessing`: protocol for `step(...)` adapter (`StepState -> StepState`).
- `StepConditionVerifying`: async condition for `with_condition`.
- `StepLoopConditionVerifying`: async `(state, iteration) -> bool` for loops.
- `StepContextMutating`: async context transformer for `Step.updating_context`.
- `StepMerging`: async merge protocol for `Step.concurrent`.
- `StepStatePreserving`: async state sink for `Step.preserving_state`.
- `StepStateRestoring`: async state loader for `Step.restoring_state`.

Example protocol-compatible callables:

```python
async def preserve_to_memory(state: StepState) -> None:
    ...


async def restore_from_memory() -> StepState:
    return StepState.of()
```

## StepException

`StepException` carries typed failure context:

- `state`: `StepState` at failure point,
- `meta`: structured metadata (`haiway.Meta`).

Example:

```python
try:
    await output_guard.process()
except StepException as exc:
    print(exc.state)
    print(exc.meta)
```

## End-To-End Advanced Pipeline

```python
class Flags(State):
    ready: bool


async def needs_more_work(*, state: StepState, iteration: int) -> bool:
    flags = state.get(Flags, default=Flags(ready=False))
    return not flags.ready and iteration < 2


base = Step.sequence(
    Step.appending_input(MultimodalContent.of("Review this deployment incident.")),
    Step.generating_completion(
        instructions="Be concise. Ask for tools only when required.",
        tools=Toolbox.empty,
    ),
    Step.handling_tools(Toolbox.empty),
    Step.updating_artifacts(Flags(ready=True)),
)

pipeline = Step.loop(
    base.with_retry(limit=1, delay=0.2).with_fallback(
        Step.appending_output(MultimodalContent.of("Unable to complete request."), emitting=True)
    ),
    condition=needs_more_work,
)

state = await pipeline.process()
content = await pipeline.run(state)
```

## Practical Notes

- `StepState` is immutable; always return new state instances.
- Wrapper order matters (`with_retry(...).with_fallback(...)` differs from reverse order).
- `run()` ignores non-multimodal chunks by design; use `stream()` if you need reasoning/tool chunks.
- `with_volatile_context()` and `with_volatile_tools()` are useful for isolation between stages.
- Use `required=True` in `StepState.get(...)` when missing artifacts should fail fast.
