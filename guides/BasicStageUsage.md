# Basic Stage Usage

Stages are the core processing pipeline units in draive that provide a composable, type-safe way to build complex LMM workflows. A Stage represents a transformation that processes an LMM context and multimodal content, allowing precise control over data flow in your applications.

## Core Concepts

### What is a Stage?

A Stage is an **immutable unit of work** that encapsulates transformations within an LMM context. Each stage:

- Receives `StageState` containing context and result
- Performs processing or transformation
- Returns updated `StageState`
- Can be chained, looped, run concurrently, or conditionally executed

## Essential Stage Methods

### 1. Completion Stages

Generate LLM completions with optional tools and output formatting:

```python
# Basic completion
basic_stage = Stage.completion(
    "Explain quantum computing",
    instruction="Provide a clear, concise explanation"
)

# Completion with tools
from draive import tool

@tool
async def calculate(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tool_stage = Stage.completion(
    "What is 15 + 27?",
    instruction="Use the calculator tool to solve this",
    tools=[calculate]
)

# Completion with specific output format
structured_stage = Stage.completion(
    "List three programming languages",
    instruction="Return as a JSON array",
    output="json"
)
```

Each completion stage will extend the context with input, generated output, and tools used. LMMOutput will be also used as the result of this stage.

### 2. Predefined Content

Insert static content into the pipeline:

```python
# Add predefined elements
static_stage = Stage.predefined(
    "System: Processing user request...",
    "User input received"
)
```

Predefined stage will extend the context with provided elements. When elements are not explicitly declared as a concrete LMMContext types, each odd element will become LMMInput and each even element will become LMMCompletion in the result context. Make sure to have equal number of inputs and outputs in the result context. Last LMMOutput will be used as the result of this stage.

### 3. Prompting Completion

Generate completions with dynamic input:

```python
async def get_user_query():
    # Simulated user input
    return "What is the weather like today?"

prompting_stage = Stage.prompting_completion(
    get_user_query,
    instruction="Answer the user's question",
    tools=[weather_tool]
)
```

### 4. Loopback Completion

Use the last completion as new input for iterative refinement:

```python
# This takes the last completion and feeds it back as input
refinement_stage = Stage.loopback_completion(
    instruction="Improve and refine the previous response",
    tools=[research_tool]
)
```

### 5. Context and Result Transformations

Modify context or result without full reprocessing:

```python
# Transform only the result
transform_result_stage = Stage.transform_result(
    lambda content: MultimodalContent.of("Transformed: ", content)
)
```
Result transformation does not change the context.

```python
# Transform only the context
transform_context_stage = Stage.transform_context(
    lambda context: context + (LMMInput.of("Additional context"),...)
)
```
Context transformation does not change the result. Make sure to have equal number of inputs and outputs in the result context.

## Stage Composition

### Looping

Execute stages repeatedly while a condition is met:

```python
async def should_continue(
    *,
    state: StageState,
    iteration: int
) -> bool:
    # Stop after 3 iterations or when result contains "done"
    return iteration < 3 and "done" not in state.result.as_string()

loop_stage = Stage.loop(
    Stage.completion("Refine the analysis further"),
    condition=should_continue,
    mode="post_check"  # Check condition after each execution
)
```

### Sequential Execution

Chain stages to run one after another:

```python
# Create individual stages
analyze_stage = Stage.completion(
    "Large dataset with user behavior patterns...",
    instruction="Analyze the input data and identify key patterns"
)

summarize_stage = Stage.completion(
    "Summarize the analysis in 2-3 sentences"
)

format_stage = Stage.completion(
    "Format the summary as a bullet-point list"
)

# Chain them together
pipeline = Stage.sequence(
    analyze_stage,
    summarize_stage,
    format_stage
)

# Execute the entire pipeline
result = await pipeline.execute()
```

### Routing

Route execution to different stages based on conditions:

```python
# Create stages with metadata for routing
analysis_stage = Stage.completion(
    "Perform detailed analysis",
    instruction="Analyze the data thoroughly"
).with_meta(
    name="detailed_analysis",
    description="Performs comprehensive data analysis"
)

summary_stage = Stage.completion(
    "Create a brief summary",
    instruction="Summarize the key points"
).with_meta(
    name="quick_summary", 
    description="Creates a quick summary of the data"
)

# Router automatically selects the appropriate stage
router_stage = Stage.router(
    analysis_stage,
    summary_stage,
    # Optional: custom routing function
    # routing=custom_routing_function
)
```

### Concurrent Execution

Run stages in parallel and merge results:

```python
async def merge_results(
    branches: Sequence[StageState | StageException]
) -> StageState:
    # Custom logic to merge multiple stage results
    successful_states = [
        state for state in branches 
        if isinstance(state, StageState)
    ]
    
    # Combine all results
    combined_content = MultimodalContent.of(
        *[state.result for state in successful_states]
    )
    
    return successful_states[0].updated(result=combined_content)

concurrent_stage = Stage.concurrent(
    Stage.completion("Analyze aspect A"),
    Stage.completion("Analyze aspect B"), 
    Stage.completion("Analyze aspect C"),
    merge=merge_results
)
```

### Conditional Execution

Execute stages based on conditions:

```python
async def needs_analysis(
    *,
    state: StageState,
) -> bool:
    # check if needs analysis...

# Stage that runs only if condition is met
conditional_stage = Stage.completion(
    "Prepare some analysis..."
).when(
    condition=needs_analysis,
    alternative=Stage.completion("Do somethin else...")
)
```

## Stage Metadata and Behavior

### Adding Metadata

Stages can have descriptive metadata which can be added when defining a stage:

```python
documented_stage = Stage.completion(
    "Process query",
    meta={
        "name": "query_processor",
        "description": "Processes and analyzes user query",
        "tags": ("processing", "nlp"),
    }
)
```

Meta can be also altered when needed by making copies of stage.

```python
documented_stage = Stage.completion(
    "Process query"
).with_meta(
    name="query_processor",
    description="Processes and analyzes user query",
    tags=("processing", "nlp")
)
```

### Caching

Add caching to avoid recomputing expensive operations:

```python
cached_stage = Stage.completion(
    "Perform complex analysis"
).cached(
    limit=10,  # Cache up to 10 results
    expiration=3600  # Cache expires after 1 hour
)
```

### Retry Logic

Add resilience with automatic retries:

```python
resilient_stage = Stage.completion(
    "Process data with external API"
).with_retry(
    limit=3,  # Retry up to 3 times
    delay=1.0,  # Wait 1 second between retries
    catching=Exception  # Retry on any exception
)
```

### Tracing

Enable execution tracing for debugging:

```python
traced_stage = Stage.completion(
    "Debug this complex operation"
).traced(label="debug_operation")
```

## Memory Integration

Stages can work with memory for context persistence:

```python
from draive.helpers import VolatileMemory

# Create memory instance
memory: Memory[LMMContext, LMMContext] = VolatileMemory(initial=())

# Stage that remembers context
remember_stage = Stage.memory_remember(memory)

# Stage that recalls previous context
recall_stage = Stage.memory_recall(memory, handling="merge")
```

## Error Handling

### Fallback Stages

Provide alternative processing on errors:

```python
primary_stage = Stage.completion(
    "Primary processing method"
)

fallback_stage = Stage.completion(
    "Fallback processing method"
)

robust_stage = primary_stage.with_fallback(
    fallback_stage,
    catching=ConnectionError  # Fall back on connection errors
)
```

### Context Trimming

Manage context size to avoid token limits:

```python
# Create a sequence that first trims context, then processes
trimmed_stage = Stage.sequence(
    Stage.trim_context(limit=4),  # Keep only last 4 context elements
    Stage.completion("Process with limited context")
)

# Strip tool-related elements from context
clean_stage = Stage.sequence(
    Stage.completion("Process with tools", tools=[some_tool]),
    Stage.strip_context_tools()  # Remove tool calls from context
)
```

### Volatile Context Management

Control what changes persist in the context:

```python
# Stage that discards context changes but keeps result
volatile_stage = Stage.completion(
    "Process but don't save context changes"
).with_volatile_context()

# Stage that discards only tool-related context changes
volatile_tools_stage = Stage.completion(
    "Use tools but don't keep tool calls in context",
    tools=[some_tool]
).with_volatile_tools_context()
```

### Result Manipulation

Control how stage results are handled:

```python
# Stage that ignores its result and keeps the previous one
ignored_stage = Stage.completion(
    "Process something but ignore the result"
).ignore_result()

# Stage that extends the current result instead of replacing it
extending_stage = Stage.completion(
    "Add more information"
).extend_result()
```

## Advanced Patterns

### Method Chaining

You can combine multiple behaviors fluently:

```python
comprehensive_stage = (
    Stage.completion("Analyze and process data")
    .with_meta(name="data_processor", description="Main data processing stage")
    .cached(limit=5)
    .with_retry(limit=2)
    .traced(label="data_processing")
    .when(lambda *, state: len(state.result.as_string()) > 10)
)
```

### Custom Stage Functions

Stages can be also created fully customized by using the `stage` decorator:

```python
from draive import stage

@stage
async def custom_processor(
    *,
    state: StageState,
) -> StageState:
    # Custom processing logic
    processed_result = MultimodalContent.of(
        f"Processed: {state.result.as_string()}"
    )
    return state.updated(result=processed_result)

# Use the custom stage as regular stage
custom_stage: Stage = custom_processor
```

## Complete Example

Here's a comprehensive example combining multiple concepts:

```python
from draive import ctx, Stage, MultimodalContent, StageState, tool
from draive.openai import OpenAI, OpenAIChatConfig
from collections.abc import Sequence
from draive.stages.types import StageException

@tool
async def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())

async def process_document(document: str):
    async with ctx.scope(
        "document_processor",
        OpenAIChatConfig(model="gpt-4o"),
        disposables=(OpenAI(),),
    ):
        # Create processing pipeline
        pipeline = Stage.sequence(
            # Step 1: Analyze document
            Stage.completion(
                document,
                instruction="Analyze the document structure and content"
            ).with_meta(name="analyzer", description="Document analysis"),

            # Step 2: Add word count if document is long
            Stage.completion(
                "Add word count information using the tool",
                tools=[word_count]
            ).when(
                condition=lambda *, state: len(state.result.as_string()) > 100
            ),

            # Step 3: Generate summary
            Stage.completion(
                "Create a concise summary"
            ).cached(limit=10)
        ).with_retry(limit=2).traced(label="document_pipeline")

        # Execute pipeline
        result = await pipeline.execute()

        return result.as_string()

# Run the example
summary = await process_document(document="...")
print(summary)
```

This guide covers the essential concepts and patterns for using Stages in draive.
