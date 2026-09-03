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

When stopping before the stream ends, close it explicitly (i.e. through `ctx.closing`).
Leaving it to garbage collection finalizes it within an unrelated context and breaks scoped state
teardown.

## Common Composition Helpers

- `Step.sequence(...)` for deterministic pipelines.
- `Step.loop(..., condition=...)` for iterative processing.
- `Step.concurrent(..., merge=...)` for fan-out/fan-in branches.
- `Step.selection(...)` for choosing the next step from the current state.
- `Step.generating_completion(...)` for one model completion stage.
- `Step.looping_completion(...)` for model + tool iterative loops.
- `Step.handling_tools(...)` for executing tool requests of the latest model output.
- `Step.appending_input(...)` / `Step.appending_output(...)` for injecting externally provided
    content, including awaited providers and `Template` values.
- `Step.appending_context(...)`, `Step.replacing_context(...)` and `Step.mutating_context(...)` for
    working with the whole `ModelContext` - append elements, swap it for a fixed one (or an awaited
    provider), or rewrite it through an async mutation.
- `Step.preserving_state(...)` / `Step.restoring_state(...)` for snapshotting whole `StepState`.
- `step` decorator for adapting a plain `StepState -> StepState` coroutine into a `Step`.

## Modifiers

Any step can be wrapped without breaking composition:

- `step.with_ctx(...)` for scoped state and disposables.
- `step.with_retry(...)` and `step.with_fallback(...)` for failure handling.
- `step.with_condition(..., alternative=...)` for conditional execution.
- `step.with_isolated_context(...)`, `step.with_volatile_context()`, `step.with_volatile_tools()`
    for controlling what the step contributes back to context.
- `step.with_suppressed_output()` for muting emitted chunks.
- `step.with_context_evaluation(...)` and `step.with_output_evaluation(...)` for evaluating context
    and emitted output.
