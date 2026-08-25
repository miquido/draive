# Stage API Status

`Stage` is not part of the current public Draive API.

Use `draive.steps.Step` instead.

## Migration mapping

- `Stage` -> `Step`
- `StageState` -> `StepState`
- `stage` decorator -> `step` decorator
- stage execution -> `run(...)`, `process(...)`, `stream(...)`

There is no `StageState.result` equivalent. A `Step` emits output chunks as it executes - collected
by `run(...)`, observed through `stream(...)` - and carries typed artifacts in
`StepState.artifacts` instead of a single result value.

Predefined stages map onto steps as follows:

- `Stage.completion(input, ...)` -> `Step.looping_completion(input=..., ...)`, or
    `Step.generating_completion(...)` when tool requests should not be resolved in a loop
- `Stage.prompting_completion(provider, ...)` -> `Step.appending_input(provider)` followed by a
    completion step
- `Stage.transform_context(...)` -> `Step.mutating_context(...)`
- `Stage.trim_context(...)` / `Stage.strip_context_tools()` -> `Step.mutating_context(...)` with an
    explicit context mutation; use `step.with_volatile_tools()` to drop tool-bearing elements a step
    produced
- `Stage.router(...)` -> `Step.selection(...)`, providing the selecting function explicitly - there
    is no built-in model-driven routing
- `Stage.memory_recall(...)` / `Stage.memory_remember(...)` -> `Step.restoring_state(...)` /
    `Step.preserving_state(...)`, or `AgentMemory.recall_step()` / `AgentMemory.remember_step()`
    within agents
- `Stage.result_evaluation(...)` / `Stage.context_evaluation(...)` ->
    `step.with_output_evaluation(...)` / `step.with_context_evaluation(...)`
- `stage.when(...)` -> `step.with_condition(...)`
- `stage.with_volatile_tools_context()` -> `step.with_volatile_tools()`
- `stage.ignore_result()` -> `step.with_suppressed_output()`

Protocol types follow the same renaming: `StageExecution` -> `StepExecuting`, `StageMerging` ->
`StepMerging`, `StageConditioning` -> `StepConditionVerifying`, `StageLoopConditioning` ->
`StepLoopConditionVerifying`, `StageContextTransforming` -> `StepContextMutating`, `StageMemory` ->
`StepStatePreserving` and `StepStateRestoring`, `StageException` -> `StepException`.

## Quick replacement example

```python
from draive.models import ModelInput
from draive.multimodal import MultimodalContent
from draive.steps import Step

pipeline = Step.sequence(
    Step.appending_context(ModelInput.of(MultimodalContent.of("Analyze this."))),
    Step.generating_completion(instructions="Answer concisely."),
)

result = await pipeline.run()
```

See [Basic Step Usage](./BasicStepUsage.md) for the active API.
