# Comprehensive Evaluation Framework

Use Draive's evaluation primitives to score model outputs consistently and keep quality criteria
transparent. This guide walks through evaluators, scenarios, suites, and supporting patterns for
building end-to-end evaluation flows.

## Evaluator Basics

- Evaluators are async callables decorated with `@evaluator` that return an `EvaluationScore` or a
  compatible numeric value.
- Thresholds determine whether an evaluation passes; named levels (`"perfect"`, `"excellent"`,
  `"good"`, `"fair"`, `"poor"`) are easier to reason about than raw floats.
- `EvaluatorResult.performance` is reported as a percentage and can exceed 100 when a score
  comfortably beats its threshold.

### Working with `EvaluationScore`

```python
from draive import EvaluationScore

score_from_float = EvaluationScore.of(0.85)
score_from_label = EvaluationScore.of("good")
score_from_boolean = EvaluationScore.of(True)
```

### Defining an evaluator

```python
from draive import evaluator

@evaluator(name="length_check", threshold="excellent")
async def check_response_length(value: str, min_length: int = 100) -> float:
    actual_length = len(value)
    if actual_length >= min_length:
        return 1.0
    return actual_length / min_length
```

```python
# Prepared evaluators freeze arguments for reuse
strict_length_check = check_response_length.prepared(min_length=200)
result = await strict_length_check("This is a test response...")
assert result.passed  # True when score >= excellent (0.7)
```

## Combining Evaluators with Scenarios

Use `evaluator_scenario` to bundle related evaluators and `evaluate` to execute them together.

```python
from collections.abc import Sequence

from draive.evaluation import evaluate, evaluator_scenario, EvaluatorResult

@evaluator_scenario(name="quality_checks")
async def evaluate_response_quality(value: str, context: str) -> Sequence[EvaluatorResult]:
    return await evaluate(
        value,
        check_response_length.prepared(),
        check_sentiment.prepared(),
        check_relevance.prepared(context=context),
        check_grammar.prepared(),
    )
```

`evaluate` can run evaluators concurrently. Limit concurrency when evaluators hit rate-limited
services.

```python
async def evaluate_response_quality_parallel(value: str, context: str) -> Sequence[EvaluatorResult]:
    return await evaluate(
        value,
        check_response_length.prepared(),
        check_sentiment.prepared(),
        check_relevance.prepared(context=context),
        check_grammar.prepared(),
        concurrent_tasks=2,
    )
```

## Evaluator Suites for Regression Testing

Suites persist test cases, run them in bulk, and expose reporting helpers.

```python
from pathlib import Path
from typing import Sequence

from draive import DataModel, evaluator_suite
from draive.evaluation import EvaluatorResult


class QATestCase(DataModel):
    question: str
    expected_topics: list[str]
    min_length: int = 100


@evaluator_suite(
    QATestCase,
    name="qa_validation",
    storage=Path("./test_cases.json"),
    concurrent_evaluations=5,
)
async def qa_test_suite(parameters: QATestCase) -> Sequence[EvaluatorResult]:
    answer = await generate_answer(parameters.question)

    return [
        await check_response_length(answer, parameters.min_length),
        await check_topic_coverage(answer, parameters.expected_topics),
        await check_factual_accuracy(answer, parameters.question),
    ]
```

```python
await qa_test_suite.add_case(
    QATestCase(
        question="What is machine learning?",
        expected_topics=["algorithms", "data", "training"],
        min_length=150,
    )
)

all_cases = await qa_test_suite.cases()
full_results = await qa_test_suite()
sample_results = await qa_test_suite(5)
partial_results = await qa_test_suite(0.3)
specific_results = await qa_test_suite(["case-1", "case-2"])

report = full_results.report(detailed=True, include_passed=False)
```

## Composing and Transforming Evaluators

```python
from draive.evaluation import Evaluator

conservative_eval = Evaluator.lowest(
    evaluator1.prepared(),
    evaluator2.prepared(),
    evaluator3.prepared(),
)
optimistic_eval = Evaluator.highest(
    evaluator1.prepared(),
    evaluator2.prepared(),
)
```

```python
# Transform inputs before delegation
field_evaluator = my_evaluator.contra_map(MyModel._.attribute.path)
normalized = my_evaluator.contra_map(lambda data: data["response"].strip().lower())
```

## Stateful Evaluation with Haiway

```python
from haiway import State, ctx

from draive import evaluator


class EvaluationConfig(State):
    strict_mode: bool = False
    max_retries: int = 3


@evaluator(threshold="perfect", state=[EvaluationConfig(strict_mode=True)])
async def strict_evaluator(value: str) -> float:
    config = ctx.state(EvaluationConfig)
    if config.strict_mode:
        # Apply stricter logic
        return await evaluate_strict(value)
    return await evaluate_lenient(value)
```

## Threshold Strategy

```python
from draive.evaluators import (
    coherence_evaluator,
    completeness_evaluator,
    consistency_evaluator,
    creativity_evaluator,
    factual_accuracy_evaluator,
    forbidden_keywords_evaluator,
    groundedness_evaluator,
    helpfulness_evaluator,
    readability_evaluator,
    required_keywords_evaluator,
    safety_evaluator,
    similarity_evaluator,
    tone_style_evaluator,
)

safety_check = safety_evaluator.with_threshold("perfect")
consistency_check = consistency_evaluator.with_threshold("perfect")
forbidden_check = forbidden_keywords_evaluator.with_threshold("perfect")

helpfulness_check = helpfulness_evaluator.with_threshold("excellent")
factual_accuracy_check = factual_accuracy_evaluator.with_threshold("excellent")
tone_style_check = tone_style_evaluator.with_threshold("excellent")

completeness_check = completeness_evaluator.with_threshold("good")
creativity_check = creativity_evaluator.with_threshold("good")
readability_check = readability_evaluator.with_threshold("good")

similarity_check = similarity_evaluator.with_threshold("fair")
keyword_check = required_keywords_evaluator.with_threshold("fair")

precise_check = factual_accuracy_evaluator.with_threshold(0.85)
```

**Threshold guidelines**

- **Safety & compliance**: use `"perfect"`; violations are unacceptable.
- **Core quality**: use `"excellent"` for user-facing content.
- **Supportive signals**: use `"good"` or lower when outcomes are subjective.

## Rich Metadata

```python
from datetime import datetime

from draive import EvaluationScore, evaluator


@evaluator
async def evaluate_with_context(response: str) -> EvaluationScore:
    score, issues = await analyze_response(response)

    return EvaluationScore(
        value=score,
        meta={
            "timestamp": datetime.now().isoformat(),
            "issues_found": issues,
            "evaluation_model": "gpt-5",
            "confidence": 0.85,
        },
    )
```

## Generating Test Cases

```python
examples = [
    TestCase(
        input="Complex technical scenario",
        expected_behavior="Detailed technical response",
        edge_cases=["unicode", "special chars", "empty input"],
    ),
    TestCase(
        input="Simple query",
        expected_behavior="Concise response",
        edge_cases=["typos", "ambiguity"],
    ),
]

cases = await suite.generate_cases(
    count=20,
    examples=examples,
    guidelines="""
    Generate diverse test cases covering:
    - Different complexity levels
    - Various input formats
    - Edge cases and error conditions
    - Performance boundaries
    """,
    persist=True,
)
```

## Summary

- Flexible scoring with normalized values and named levels
- Composable evaluators with thresholds and metadata
- Scenario grouping for related checks
- Suite management with persistent storage and generation tools
- Reporting helpers for insight into failures and regressions
- Concurrent execution to balance latency and throughput

Use evaluators for quick checks, scenarios for logical groupings, and suites for comprehensive
regression coverage backed by persistent cases and automated generation.
