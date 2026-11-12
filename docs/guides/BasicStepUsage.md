# Basic Step Usage

`Step` is Draive's composable pipeline primitive. It transforms immutable `StepState` and can emit
streamed output chunks.

A step pipeline can both:

- mutate state (context + artifacts), and
- emit output chunks for streaming consumers.

## Execution Modes

- `await step.run(...)` collects emitted multimodal parts into `MultimodalContent`.
- `await step.process(...)` returns final `StepState`.
- `async for chunk in step.stream(...)` yields non-state output chunks.

Choose mode based on what your caller needs: final content, final state, or incremental output.

## Minimal Example

```python
from draive import State
from draive.models import ModelInput
from draive.multimodal import MultimodalContent
from draive.steps import Step, StepState


class Flags(State):
    ready: bool


pipeline = Step.sequence(
    # Add user input to model context.
    Step.appending_context(ModelInput.of(MultimodalContent.of("Analyze this input."))),
    # Emit user-visible chunk.
    Step.emitting("Working..."),
    # Persist typed artifacts for downstream steps.
    Step.updating_artifacts(Flags(ready=True), status=Flags(ready=True)),
)

# Content-focused mode.
content = await pipeline.run()
# State-focused mode.
state: StepState = await pipeline.process()
```

## Access Artifacts

Artifacts are stored by type name (or custom key when provided as keyword).

```python
flags = state.get(Flags, required=True)
status = state.get(Flags, key="status", required=True)
```

## Stream Output

```python
async for chunk in pipeline.stream():
    print(chunk)
```

When stopping before the stream ends, close it explicitly (i.e. through `ctx.closing`). Leaving it
to garbage collection finalizes it within an unrelated context and breaks scoped state teardown.

## Common Composition Helpers

- `Step.sequence(...)` for deterministic pipelines, also available as `first.then(second)` or
    `first >> second`.
- `Step.loop(..., condition=...)` for iterative processing. `Step.loop` and
    `Step.looping_completion` stop with a `StepException` after `max_iterations` (32 by default,
    `None` disables the limit).
- `Step.concurrent(..., merge=...)` for fan-out/fan-in branches.
- `Step.selection(...)` for choosing the next step from the current state.
- `Step.generating_completion(...)` for one model completion stage.
- `Step.looping_completion(...)` for model + tool iterative loops.
- `Step.decoding_artifact(...)` for decoding structured model output into an artifact.
- `Step.handling_tools(...)` for executing tool requests of the latest model output.
- `Step.appending_input(...)` / `Step.appending_output(...)` for injecting externally provided
    content, including awaited providers and `Template` values.
- `Step.appending_context(...)`, `Step.replacing_context(...)` and `Step.mutating_context(...)` for
    working with the whole `ModelContext` - append elements, swap it for a fixed one (or an awaited
    provider), or rewrite it through an async mutation.
- `Step.preserving_state(...)` / `Step.restoring_state(...)` for snapshotting whole `StepState`.
- `Step.updating_artifacts(...)` / `Step.removing_artifacts(...)` for changing artifacts.
- `step` decorator for adapting a plain `StepState -> StepState` coroutine into a `Step`.

## Structured Output

Request typed output from a completion and decode it into an artifact with
`Step.decoding_artifact(...)`:

```python
report = await (
    Step.looping_completion(instructions="Summarize...", output=Report)
    >> Step.decoding_artifact(Report)
).process(context)
report.get(Report, required=True)
```

Use `Step.removing_artifacts(...)` or `StepState.removing_artifacts(...)` to drop artifacts that
should not be passed further.

## Modifiers

Any step can be wrapped without breaking composition:

- `step.with_ctx(...)` for scoped state and disposables.
- `step.with_retry(...)` and `step.with_fallback(...)` for failure handling. Every attempt (and the
    fallback) starts from the original input state, partial state of a failed attempt is discarded.
    Output chunks already emitted cannot be retracted.
- `step.with_timeout(seconds)` limiting step execution time, raising `StepTimeout`. Time the
    consumer spends between emitted chunks is not counted.
- `step.with_condition(..., alternative=...)` for conditional execution.
- `step.with_isolated_context(...)`, `step.with_volatile_context()`, `step.with_volatile_tools()`
    for controlling what the step contributes back to context.
- `step.with_suppressed_output()` for muting emitted chunks.
- `step.with_context_evaluation(...)` and `step.with_output_evaluation(...)` for evaluating context
    and emitted output. Output evaluation runs once, after the wrapped step completes, over all
    emitted chunks.

## Failures and Limits

Step failures carry the `StepState` they happened with (`exception.state`):

| Exception              | Raised when                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `StepException`        | evaluation fails with `raise_on_failure`, decoding fails, or a loop exceeds `max_iterations`        |
| `StepTimeout`          | `with_timeout(...)` limit is exceeded (`exception.timeout`)                                         |
| `StepRetriesExhausted` | `with_retry(...)` failed after at least one retry (`exception.attempts`, last error as `__cause__`) |

Errors that are not matched by `catching`, or that happen with `limit=0`, are raised unchanged.

## Artifacts

Artifacts given to `StepState.of`, `StepState.updating_artifacts` or `Step.updating_artifacts` must
resolve to unique keys within a single call (a `ValueError` is raised otherwise); artifacts already
stored in state are replaced.

`with_volatile_tools()` preserves the context passed into the wrapped step. In context added or
replaced by the step, it removes tool request and response blocks while retaining any ordinary
content stored in the same context elements.

## Extending Steps

The callable protocols used by `Step` are public when custom orchestration needs explicit type
annotations. Import them from `draive` (or `draive.steps`):

```python
from draive import StepExecuting, StepMerging, StepProcessing, StepSelecting
```

`StepExecuting` describes the low-level streaming interface accepted by `Step(...)`.
`StepProcessing` describes state-only functions accepted by the `@step` decorator. The remaining
protocols describe the callbacks accepted by composition and modifier methods.
