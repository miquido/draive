# Basic Stage Usage

Stages are the core processing pipeline units in draive that provide a composable, type-safe way to build complex LMM workflows. A Stage represents a transformation that processes an LMM context and multimodal content, allowing precise control over data flow in your applications.

## Core Concepts

### What is a Stage?

A Stage is an **immutable unit of work** that encapsulates transformations within an LMM context. Each stage:

- Receives `StageState(context: LMMContext, result: MultimodalContent)`
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

### 3. Context and Result Transformations

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

### Conditional Execution

Execute stages based on conditions:

```python
def needs_analysis(
    *,
    meta: Meta,
    context: LMMContext,
    result: MultimodalContent,
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
remember_stage = Stage.completion(
    "Process and remember this information"
).memory_remember(memory)

# Stage that recalls previous context
recall_stage = Stage.completion(
    "Use previous context to answer"
).memory_recall(memory, handling="append")
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
trimmed_stage = Stage.completion(
    "Process with limited context"
).trim_context(limit=4)  # Keep only last 4 context elements
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
    .when(lambda state: len(state.result.as_string()) > 10)
)
```

### Custom Stage Functions

Stages can be also created fully customized by using the `stage` decorator:

```python
from draive import stage

@stage
async def custom_processor(
    *,
    context: LMMContext,
    result: MultimodalContent
) -> StageState:
    # Custom processing logic
    processed_result = MultimodalContent.of(
        f"Processed: {result.as_string()}"
    )
    return StageState(context=context, result=processed_result)

# Use the custom stage as regular stage
custom_stage: Stage = custom_processor
```

## Complete Example

Here's a comprehensive example combining multiple concepts:

```python
from draive import ctx, Stage, MultimodalContent, tool
from draive.openai import OpenAI, OpenAIChatConfig

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
                condition=lambda state: len(state.result.as_string()) > 100
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
