# Basic Stage Usage

Stages are the programmable building blocks of Draive pipelines. Every `Stage` receives a
`StageState` (which encapsulates the active context and latest result), performs its work, and
returns a new state. This guide walks through the most common operations so you can assemble
reliable pipelines with confidence.

## Core Concepts

- **Stage** – an async callable that produces or transforms multimodal content.
- **StageState** – an immutable snapshot holding context entries and the latest result. Always
    return `state.updating(...)` instead of mutating in place.
- **MultimodalContent** – container for text, images, artifacts, and resources used as model inputs
    or outputs.
- **`ctx.scope(...)`** – binds providers, disposables, and logging/metrics for the lifetime of your
    pipeline. All observability flows through `ctx`.

## Creating Your First Completion

```python
from draive import Stage

basic_stage = Stage.completion(
    "Explain quantum computing",
    instructions="Keep the answer concise and friendly.",
)

result_state = await basic_stage.execute()
summary = result_state.result.as_string()
```

`Stage.completion` wires a model call, applies optional instructions, and returns a `StageState`
whose `result` contains the generated content. Use `.execute()` for quick experiments or tests; in
production you typically compose stages into larger flows.

## Adding Tools and Structured Output

```python
from draive import Stage, tool

@tool
async def calculate(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tooling_stage = Stage.completion(
    "What is 15 + 27?",
    instructions="Use the provided calculator tool.",
    tools=[calculate],
)

structured_stage = Stage.completion(
    "List three programming languages",
    instructions="Return a JSON array of language names.",
    output="json",
)
```

Provide the tools the model is allowed to call, and use `output` when you need structured responses
(for example, `"json"` or `"yaml"`).

## Working with Static or Prompted Content

```python
from draive import Stage

static_stage = Stage.predefined(
    "System: Processing user request...",
    "User input received.",
)

async def get_user_query() -> str:
    return "What's the weather like today?"

prompted_stage = Stage.prompting_completion(
    get_user_query,
    instructions="Answer clearly and reference the city if provided.",
)

loopback_stage = Stage.loopback_completion(
    instructions="Polish the previous response for clarity.",
)
```

`Stage.predefined` injects fixed conversation turns. Prompting and loopback stages fetch new input
at runtime and feed it into subsequent completions.

## Transforming Context and Results

```python
from draive import ModelInput, MultimodalContent, Stage

transform_result_stage = Stage.transform_result(
    lambda content: MultimodalContent.of("Transformed: ", content),
)

transform_context_stage = Stage.transform_context(
    lambda context: context + (ModelInput.of("Additional context"),),
)
```

Use transformers when you need to adjust only the result or the conversation context. Because
`StageState` is immutable, each transformation returns a fresh copy.

## Looping Until a Condition Is Met

```python
from draive import Stage, StageState

async def should_continue(*, state: StageState, iteration: int) -> bool:
    return iteration < 3 and "done" not in state.result.as_string().lower()

loop_stage = Stage.loop(
    Stage.completion("Refine the analysis further."),
    condition=should_continue,
    mode="post_check",  # Evaluate the condition after each iteration
)
```

`Stage.loop` repeatedly executes a stage while `condition` stays true. The `iteration` argument lets
you cap retries or break on custom signals.

## Sequencing Stages into Pipelines

```python
from draive import Stage

analyze_stage = Stage.completion(
    "User behaviour dataset...",
    instructions="Identify key engagement patterns.",
)

summarize_stage = Stage.completion(
    "Summarize the analysis in 2-3 sentences.",
)

format_stage = Stage.completion(
    "Format the summary as bullet points.",
)

pipeline = Stage.sequence(
    analyze_stage,
    summarize_stage,
    format_stage,
)

final_state = await pipeline.execute()
```

`Stage.sequence` runs each stage in order, feeding the updated state forward. You can nest sequences
to build larger flows.

## Routing, Concurrency, and Merging

```python
from draive import Stage

analysis_stage = Stage.completion(
    "Perform detailed analysis.",
    instructions="Go deep on the data.",
).with_meta(
    name="detailed_analysis",
    description="Full analysis of the input dataset.",
)

summary_stage = Stage.completion(
    "Create a brief summary.",
    instructions="Highlight only the top three insights.",
).with_meta(
    name="quick_summary",
    description="Lightweight overview for dashboards.",
)

router_stage = Stage.router(
    analysis_stage,
    summary_stage,
    # routing=custom_router,  # Optionally provide your own routing logic
)
```

```python
from collections.abc import Sequence

from draive import MultimodalContent, Stage, StageState
from draive.stages.types import StageException

async def merge_results(
    branches: Sequence[StageState | StageException],
) -> StageState:
    successful = [branch for branch in branches if isinstance(branch, StageState)]
    combined = MultimodalContent.of(*[state.result for state in successful])
    return successful[0].updating(result=combined)

concurrent_stage = Stage.concurrent(
    Stage.completion("Analyze aspect A."),
    Stage.completion("Analyze aspect B."),
    Stage.completion("Analyze aspect C."),
    merge=merge_results,
)
```

Routers pick the most appropriate stage at runtime, while `Stage.concurrent` fans out work and
merges the resulting states.

## Conditional, Cached, and Resilient Stages

```python
from draive import Stage, StageState

async def needs_analysis(*, state: StageState) -> bool:
    return "analysis" not in state.result.as_string().lower()

conditional_stage = Stage.completion("Prepare detailed analysis...").when(
    condition=needs_analysis,
    alternative=Stage.completion("Skip analysis and move on."),
)

cached_stage = Stage.completion("Perform complex analysis.").cached(
    limit=10,
    expiration=3600,
)

resilient_stage = Stage.completion("Call external API.").with_retry(
    limit=3,
    delay=1.0,
    catching=Exception,
)
```

`when` toggles execution dynamically, `.cached(...)` stores results, and `.with_retry(...)` protects
against transient failures. Provide either a single exception type or a predicate such as
`lambda exc: isinstance(exc, (TimeoutError, ConnectionError))` via `catching` to control which
errors trigger a retry.

## Advanced Context Management

```python
from draive import Stage, tool

@tool
async def dummy_tool(text: str) -> str:
    return text.upper()

trimmed_stage = Stage.sequence(
    Stage.trim_context(limit=4),
    Stage.completion("Process with limited context."),
)

clean_stage = Stage.sequence(
    Stage.completion("Process with tools.", tools=[dummy_tool]),
    Stage.strip_context_tools(),
)

volatile_stage = Stage.completion(
    "Process data without persisting context.",
).with_volatile_context()

ignored_stage = Stage.completion(
    "Generate intermediate data.",
).ignore_result()

extending_stage = Stage.completion(
    "Append additional insight.",
).extend_result()
```

These helpers trim, clean, or discard context and results so downstream stages only see what they
need.

## Building Composite Behaviour

```python
from draive import Stage

composite_stage = (
    Stage.completion("Analyze and process data.")
    .with_meta(name="data_processor", description="Main processing stage")
    .cached(limit=5)
    .with_retry(limit=2)
    .traced(label="data_processing")
    .when(lambda *, state: len(state.result.as_string()) > 10)
)
```

You can chain modifiers to produce sophisticated behaviour from a single stage definition.

## Defining Custom Stages

```python
from draive import MultimodalContent, StageState, stage

@stage
async def custom_processor(*, state: StageState) -> StageState:
    processed = MultimodalContent.of(f"Processed: {state.result.as_string()}")
    return state.updating(result=processed)
```

Any async function decorated with `@stage` gains the same API as built-in stages and can be mixed
freely with them.

## End-to-End Example

```python
from draive import ctx, Stage, tool
from draive.openai import OpenAI, OpenAIResponsesConfig

@tool
async def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())

async def process_document(document: str) -> str:
    async with ctx.scope(
        "document_processor",
        OpenAIResponsesConfig(model="gpt-5"),
        disposables=(OpenAI(),),
    ):
        pipeline = Stage.sequence(
            Stage.completion(
                document,
                instructions="Analyze the document structure and content.",
            ).with_meta(name="analyzer", description="Document analysis"),
            Stage.completion(
                "Add word count information using the tool.",
                tools=[word_count],
            ).when(
                condition=lambda *, state: len(state.result.as_string()) > 100,
            ),
            Stage.completion("Create a concise summary.").cached(limit=10),
        ).with_retry(limit=2).traced(label="document_pipeline")

        result_state = await pipeline.execute()
        ctx.log_info(
            "Generated document summary",
            summary=result_state.result.as_string(),
        )
        return result_state.result.as_string()
```

This pipeline analyzes a document, augments it with tool output when needed, summarizes the result,
and records the outcome for observability. Combine these patterns to compose the exact behaviour
your application needs.
