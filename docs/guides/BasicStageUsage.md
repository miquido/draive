# Stage API Status

`Stage` is not part of the current public Draive API.

Use `draive.steps.Step` instead.

## Migration mapping

- `Stage` -> `Step`
- `StageState` -> `StepState`
- `stage` decorator -> `step` decorator
- stage execution -> `run(...)`, `process(...)`, `stream(...)`

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
