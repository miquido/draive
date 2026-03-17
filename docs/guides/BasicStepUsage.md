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

## Common Composition Helpers

- `Step.sequence(...)` for deterministic pipelines.
- `Step.loop(..., condition=...)` for iterative processing.
- `Step.concurrent(..., merge=...)` for fan-out/fan-in branches.
- `Step.generating_completion(...)` for one model completion stage.
- `Step.looping_completion(...)` for model + tool iterative loops.
