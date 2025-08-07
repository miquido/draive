# Comprehensive Evaluation Framework

The Draive evaluation framework provides a systematic approach to evaluating AI model outputs, workflows, and applications. It offers multiple levels of evaluation from individual checks to comprehensive test suites with automated case generation.

## Overview

The evaluation framework consists of three main components:

1. **Evaluators** - Individual evaluation functions with configurable thresholds
2. **Scenarios** - Groups of evaluators for comprehensive testing
3. **Suites** - Test suite management with case generation and storage

## Core Concepts

### Evaluation Scores

All evaluations produce normalized scores between 0 and 1:

```python
from draive import EvaluationScore

# Create scores from various formats
score1 = EvaluationScore.of(0.85)  # Direct float
score2 = EvaluationScore.of("good")  # Named levels
score3 = EvaluationScore.of(True)  # Boolean (1.0 for True, 0.0 for False)

# Named score levels - prefer these over numeric values for better readability
# "none" = 0.0, "poor" = 0.1, "fair" = 0.3 "good" = 0.5, "excellent" = 0.7, "perfect" = 0.9, "max" = 1.0
```

### Performance Metrics

Performance is calculated as `(score / threshold) * 100%`:
- Values can exceed 100% when score > threshold
- Aggregate calculations cap individual performances at 100% for consistency
- Normalized with up to 100% for aggregation

## Creating Evaluators

### Basic Evaluator

```python
from draive import evaluator, EvaluationScore

@evaluator(name="length_check") # custom name, default is function name
async def check_response_length(value: str, min_length: int = 100) -> float:
    """Evaluate if response meets minimum length requirements."""
    actual_length = len(value)
    if actual_length >= min_length:
        return 1.0
    return actual_length / min_length # return score directly, must be a value between 0.0 and 1.0

# Use the evaluator
result = await check_response_length("This is a test response...")
print(f"Passed: {result.passed}")  # True if score >= "excellent" (0.7)
print(f"Performance: {result.performance:.1f}%")
```

### Evaluator with Metadata

```python
@evaluator(threshold="good") # custom threshold, default is 1 (max)
async def check_sentiment(value: str) -> EvaluationScore:
    """Evaluate text sentiment using an LLM."""
    sentiment_score = await analyze_sentiment(value)

    return EvaluationScore(
        value=sentiment_score,
        meta={ # return score with additional metadata
            "sentiment": "positive" if sentiment_score > 0.5 else "negative",
            "confidence": 0.95
        }
    )
```

### Prepared Evaluators

Pre-bind arguments for reusable configurations:

```python
# Create a prepared evaluator with fixed parameters
strict_length_check = check_response_length.prepared(min_length=200)

# Use it multiple times
result1 = await strict_length_check("Response 1...")
result2 = await strict_length_check("Response 2...")
```

## Evaluator Scenarios

Scenarios group multiple evaluators for comprehensive testing:

```python
from collections.abc import Sequence
from draive.evaluation import evaluate, evaluator_scenario, EvaluatorResult

@evaluator_scenario(name="quality_checks")
async def evaluate_response_quality(
    value: str,
    context: str
) -> Sequence[EvaluatorResult]:
    """Run multiple quality checks on a response."""
    return await evaluate(
        value,
        check_response_length.prepared(),
        check_sentiment.prepared(),
        check_relevance.prepared(context=context),
        check_grammar.prepared(),
    )

# Run the evaluation
evaluation_results = await evaluate_response_quality(
    response="The model's response...",
    context="Original question context"
)

# Process results
all_passed = all(result.passed for result in evaluation_results)
avg_performance = sum(result.performance for result in evaluation_results) / len(evaluation_results)

print(f"All evaluations passed: {all_passed}")
print(f"Average performance: {avg_performance:.1f}%")

for result in evaluation_results:
    print(f"- {result.evaluator}: {result.score.value} ({'✓' if result.passed else '✗'})")
```

### Concurrent Evaluator Execution

For better performance, run evaluators concurrently using the `evaluate` helper:

```python
from draive.evaluation import evaluate

async def evaluate_response_quality_parallel(
    value: str,
    context: str
) -> Sequence[EvaluatorResult]:
    """Run multiple quality checks concurrently for better performance."""

    # Execute all evaluators concurrently using the evaluate helper
    return await evaluate(
        value,
        check_response_length.prepared(),
        check_sentiment.prepared(),
        check_relevance.prepared(context=context),
        check_grammar.prepared(),
        concurrent_tasks=2  # Run up to 2 evaluators in parallel
    )
```

**Tip**: Concurrent execution is especially beneficial when evaluators make network calls (e.g., to LLMs) or perform I/O operations. The `concurrent_tasks` parameter controls the maximum number of evaluators running simultaneously.

## Evaluation Suites

Suites provide comprehensive test management with storage and case generation:

### Creating a Suite

```python
from draive import evaluator_suite, DataModel
from pathlib import Path

class QATestCase(DataModel):
    question: str
    expected_topics: list[str]
    min_length: int = 100

@evaluator_suite(
    QATestCase,
    name="qa_validation",
    storage=Path("./test_cases.json"),  # Persistent storage
    concurrent_evaluations=5  # Run 5 cases in parallel
)
async def qa_test_suite(
    parameters: QATestCase
) -> Sequence[EvaluatorResult]:
    """Evaluate question-answer pairs."""
    # Generate answer using your QA system
    answer = await generate_answer(parameters.question)

    # Run evaluations
    return [
        await check_response_length(answer, parameters.min_length),
        await check_topic_coverage(answer, parameters.expected_topics),
        await check_factual_accuracy(answer, parameters.question)
    ]
```

### Managing Test Cases

```python
# Add test cases manually
await qa_test_suite.add_case(
    QATestCase(
        question="What is machine learning?",
        expected_topics=["algorithms", "data", "training"],
        min_length=150
    )
)

# Generate cases automatically using LLM
generated_cases = await qa_test_suite.generate_cases(
    count=10,
    persist=True,  # Save to storage
    guidelines="Focus on technical questions about AI and ML",
    examples=[  # Provide examples for better generation
        QATestCase(
            question="How does gradient descent work?",
            expected_topics=["optimization", "loss", "parameters"]
        )
    ]
)

# List all cases
all_cases = await qa_test_suite.cases()
print(f"Total cases: {len(all_cases)}")

# Remove a case
await qa_test_suite.remove_case("case-id-123")
```

### Running Suite Evaluations

```python
# Run all test cases
full_results = await qa_test_suite()

# Run specific number of random cases
sample_results = await qa_test_suite(5)

# Run percentage of cases
partial_results = await qa_test_suite(0.3)  # 30% of cases

# Run specific cases by ID
specific_results = await qa_test_suite(["case-1", "case-2"])

# Generate comprehensive report
print(
    full_results.report(
        detailed=True,
        include_passed=False  # Show only failures
    )
)
```

## Advanced Features

### Composition and Transformation

```python
# Compose evaluators - return lowest/highest score
conservative_eval = Evaluator.lowest(
    evaluator1.prepared(),
    evaluator2.prepared(),
    evaluator3.prepared()
)

optimistic_eval = Evaluator.highest(
    evaluator1.prepared(),
    evaluator2.prepared()
)

# Transform inputs before evaluation

# Extract specific field from complex object using AttributePath
field_evaluator = my_evaluator.contra_map(MyModel._.attribute.path)

# Apply custom transformation
transformed = my_evaluator.contra_map(
    lambda data: data["response"].strip().lower()
)
```

### State Management

Include additional state in evaluation context:

```python
from draive import State

class EvaluationConfig(State):
    strict_mode: bool = False
    max_retries: int = 3

@evaluator(
    threshold="perfect",  # Named threshold for clarity
    state=[EvaluationConfig(strict_mode=True)]
)
async def strict_evaluator(value: str) -> float:
    config = ctx.state(EvaluationConfig)
    if config.strict_mode:
        # Apply stricter evaluation logic
        ...
```

## Best Practices

### 1. Threshold Selection

Choose appropriate thresholds based on criticality and business impact:

```python
from draive.evaluators import *

# Critical features - highest threshold (perfect = 0.9)
safety_check = safety_evaluator.with_threshold("perfect")
consistency_check = consistency_evaluator.with_threshold("perfect")
forbidden_check = forbidden_keywords_evaluator.with_threshold("perfect")

# Important features - high threshold (excellent = 0.7)
helpfulness_check = helpfulness_evaluator.with_threshold("excellent")
factual_accuracy_check = factual_accuracy_evaluator.with_threshold("excellent")
tone_style_check = tone_style_evaluator.with_threshold("excellent")

# Quality features - moderate threshold (good = 0.5)
completeness_check = completeness_evaluator.with_threshold("good")
creativity_check = creativity_evaluator.with_threshold("good")
readability_check = readability_evaluator.with_threshold("good")

# Flexible features - lower threshold (fair = 0.3)
similarity_check = similarity_evaluator.with_threshold("fair")
keyword_check = required_keywords_evaluator.with_threshold("fair")

# Custom precise thresholds when needed
precise_check = factual_accuracy_evaluator.with_threshold(0.85)  # Between excellent (0.7) and perfect (0.9)
```

**Threshold Guidelines by Use Case:**
- **Safety & Compliance**: Always use "perfect" - no tolerance for violations
- **Core Quality**: Use "excellent" - high standards for user-facing content
- **Feature Quality**: Use "good" - balanced standards allowing some flexibility
- **Experimental/Optional**: Use "fair" - minimum acceptable standards

### 2. Meaningful Metadata

Include context in evaluation results:

```python
@evaluator
async def evaluate_with_context(response: str) -> EvaluationScore:
    score, issues = await analyze_response(response)

    return EvaluationScore(
        value=score,
        meta={
            "timestamp": datetime.now().isoformat(),
            "issues_found": issues,
            "evaluation_model": "gpt-4",
            "confidence": 0.85
        }
    )
```

### 3. Effective Case Generation

Provide good examples for LLM-based generation:

```python
# Define diverse, high-quality examples
examples = [
    TestCase(
        input="Complex technical scenario",
        expected_behavior="Detailed technical response",
        edge_cases=["unicode", "special chars", "empty input"]
    ),
    TestCase(
        input="Simple query",
        expected_behavior="Concise response",
        edge_cases=["typos", "ambiguity"]
    )
]

# Generate with clear guidelines
cases = await suite.generate_cases(
    count=20,
    examples=examples,
    guidelines="""
    Generate diverse test cases covering:
    - Different complexity levels
    - Various input formats
    - Edge cases and error conditions
    - Performance boundaries
    """
)
```

## Summary

The Draive evaluation framework provides:

- **Flexible scoring** with normalized values and named levels
- **Composable evaluators** with thresholds and metadata
- **Scenario grouping** for comprehensive testing
- **Suite management** with storage and case generation
- **Performance tracking** with detailed reporting
- **Concurrent execution** for efficient evaluation

Use evaluators for quick checks, scenarios for related validations, and suites for comprehensive testing with persistent test cases and automated generation.
