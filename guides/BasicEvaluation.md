# Basic Evaluation Usage

Draive framework provides comprehensive evaluation capabilities to assess LLM outputs and conversational flows. The evaluation system consists of three main components: individual evaluators, scenarios that combine multiple evaluators, and evaluation suites for systematic testing.

## Simple Evaluators

The simplest way to evaluate content is using individual evaluators. Let's start with a basic custom evaluator that checks if text contains specific keywords:

```python
from draive.evaluation import evaluator, EvaluationScore
from draive.multimodal import Multimodal

@evaluator(name="keyword_presence", threshold=0.8)
async def keyword_evaluator(
    content: Multimodal,
    /,
    required_keywords: list[str],
) -> EvaluationScore:
    text = str(content).lower()
    found_keywords = sum(1 for keyword in required_keywords if keyword.lower() in text)

    if not required_keywords:
        return EvaluationScore(
            value=0,
            comment="No keywords provided for evaluation",
        )

    score = found_keywords / len(required_keywords)
    return EvaluationScore(
        value=score,
        comment=f"Found {found_keywords}/{len(required_keywords)} required keywords",
    )
```

Using this evaluator is straightforward:

```python
from draive import ctx, load_env
from draive.openai import OpenAI, OpenAIChatConfig

load_env()

async with ctx.scope(
    "evaluation_example",
    OpenAIChatConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
):
    content = "AI and machine learning are transforming technology"

    result = await keyword_evaluator(
        content,
        required_keywords=["AI", "machine learning", "technology"],
    )

    print(f"Score: {result.score.value}")
    print(f"Passed: {result.passed}")
    print(f"Comment: {result.score.comment}")
```

## Built-in Evaluators

Draive includes several pre-built evaluators for common use cases. Let's explore groundedness and readability evaluators:

```python
from draive.evaluators import groundedness_evaluator, readability_evaluator

# Evaluate if generated content is grounded in source material
reference_text = """
Climate change is causing rising sea levels globally.
Scientific data shows ocean levels have risen 8-9 inches since 1880.
"""

generated_text = """
Based on scientific evidence, global sea levels have increased
approximately 8-9 inches since 1880 due to climate change impacts.
"""

groundedness_result = await groundedness_evaluator(
    generated_text,
    reference=reference_text,
)

print(f"Groundedness: {groundedness_result.score.value}")
print(f"Comment: {groundedness_result.score.comment}")

# Evaluate text readability
complex_text = """
The utilization of sophisticated methodological approaches in the
implementation of artificial intelligence systems necessitates comprehensive
understanding of underlying algorithmic paradigms.
"""

readability_result = await readability_evaluator(complex_text)

print(f"Readability: {readability_result.score.value}")
print(f"Comment: {readability_result.score.comment}")
```

## Evaluation Scenarios

Scenarios combine multiple evaluators to assess content from different perspectives. Here's a scenario that evaluates content quality using both groundedness and readability:

```python
from draive.evaluation import evaluation_scenario, EvaluationScenarioResult
from draive.evaluators import conciseness_evaluator

@evaluation_scenario(name="content_quality")
async def content_quality_scenario(
    content: str,
    /,
    *,
    reference: str,
) -> EvaluationScenarioResult:
    # Prepare evaluators with appropriate thresholds
    conciseness = conciseness_evaluator.with_threshold("excellent")
    readability = readability_evaluator.with_threshold("good")

    # Evaluate using multiple criteria
    return await EvaluationScenarioResult.evaluating(
        content,
        conciseness.prepared(reference=reference),
        readability.prepared(),
    )

# Use the scenario
scenario_result = await content_quality_scenario(
    generated_text,
    reference=reference_text,
)

print(f"Scenario passed: {scenario_result.passed}")
print(f"Overall score: {scenario_result.relative_score:.2f}")

for evaluation in scenario_result.evaluations:
    print(f"- {evaluation.evaluator}: {evaluation.score.value:.2f} ({'✓' if evaluation.passed else '✗'})")
```

## Evaluation Suites

Evaluation suites allow systematic testing across multiple test cases. Let's create a suite to evaluate different content generation scenarios:

```python
from typing import Sequence
from draive.evaluation import evaluation_suite, EvaluationSuiteCase
from draive import TextGeneration, DataModel

class ContentTestCase(DataModel):
    topic: str
    required_keywords: Sequence[str]
    reference_material: str

@evaluation_suite(ContentTestCase)
async def content_generation_suite(
    parameters: ContentTestCase,
) -> EvaluationCaseResult:
    # Generate content based on test case parameters
    content: str = await TextGeneration.generate(
        instruction=f"Write informative content about {parameters.topic}",
        input=parameters.reference_material,
    )
    return await EvaluationCaseResult.evaluating(
        content,
        content_quality_scenario.prepared(
            reference=parameters.reference_material,
        ),
        keyword_evaluator.with_threshold(0.5).prepared(
            required_keywords=parameters.required_keywords
        ),
    )

# Define test cases
test_cases = [
    ContentTestCase(
        topic="climate change",
        required_keywords=["temperature", "emissions", "global"],
        reference_material="Global temperatures have risen 1.1°C since pre-industrial times",
    ),
    ContentTestCase(
        topic="renewable energy",
        required_keywords=["solar", "sustainable", "energy"],
        reference_material="Solar and wind power are leading renewable energy sources",
    ),
]

# Prepare suite with in-memory test cases storage
suite = content_generation_suite.with_storage(test_cases)

# Execute suite evaluation
suite_results = await suite()

print(f"Suite passed: {suite_results.passed}")
print(f"Cases passed: {sum(1 for case in suite_results.cases if case.passed)}/{len(suite_results.cases)}")

for case_result in suite_results.cases:
    print(f"\nCase {case_result.case.parameters.topic}:")
    print(f"  Generated: {case_result.value[:100]}...")
    print(f"  Passed: {case_result.passed}")
    print(f"  Score: {case_result.relative_score:.2f}")
```

## Advanced Usage

You can customize evaluators with execution contexts and metadata:

```python
# Create evaluator with custom execution context
custom_evaluator = keyword_evaluator.with_execution_context(
    ctx.scope("custom_evaluation")
).with_meta({
    "version": "1.0",
    "author": "evaluation_team",
})

# Combine evaluators using logical operations
best_evaluator = Evaluator.highest(
    conciseness_evaluator.prepared(reference=reference_text),
    readability_evaluator.prepared(),
)

# Map evaluator to work with different data structures
from draive.parameters import DataModel

class DocumentContent(DataModel):
    title: str
    body: str

document_evaluator = readability_evaluator.contra_map(
    lambda doc: doc.body  # Extract body text for evaluation
)
```

The evaluation system integrates seamlessly with draive's context management and provides detailed metrics logging for comprehensive analysis of your LLM applications.
