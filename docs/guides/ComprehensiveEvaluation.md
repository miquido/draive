# Comprehensive Evaluation Framework

Use Draive's evaluation primitives to score model outputs consistently and keep quality criteria
transparent. This guide walks through evaluators, scenarios, suites, and supporting patterns for
building end-to-end evaluation flows.

## Evaluator Basics

- Evaluators are async callables decorated with `@evaluator` that return an `EvaluationScore` or a
    compatible score value (`float`, `bool`, or a level name).
- Thresholds determine whether an evaluation passes; named levels are easier to reason about than
    raw floats: `"none"` (0.0), `"poor"` (0.2), `"fair"` (0.4), `"good"` (0.6), `"excellent"` (0.8),
    `"perfect"` (1.0). The default threshold is `1` - the strictest one.
- `EvaluatorResult` carries `score`, `threshold`, `meta` and derives `passed` (`score >= threshold`).
- `EvaluatorResult.performance` is reported as a percentage and can exceed 100 when a score
    comfortably beats its threshold.
- An evaluator that raises is not propagated: the failure is logged and turned into a `0.0` score
    with the exception recorded in the result metadata.

### Working with `EvaluationScore`

```python
from draive.evaluation import EvaluationScore

score_from_float = EvaluationScore.of(0.85)
score_from_label = EvaluationScore.of("good")
score_from_boolean = EvaluationScore.of(True)
score_with_meta = EvaluationScore.of(0.85, meta={"comment": "minor omissions"})
```

`EvaluationScore` holds only `value` and `meta` - comments and any other explanation belong in
`meta`, conventionally under the `"comment"` key.

### Defining an evaluator

```python
from draive.evaluation import evaluator

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
assert result.passed  # True when score >= excellent (0.8)
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

The scenario definition returns a sequence of `EvaluatorResult`, while calling the scenario returns
an `EvaluatorScenarioResult` exposing `scenario`, `results`, `passed`, `performance` and
`report(...)`. Scenarios and plain evaluators can be mixed in one `evaluate(...)` call and in one
suite definition.

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
from collections.abc import Sequence
from pathlib import Path

from draive import State
from draive.evaluation import evaluator_suite
from draive.evaluation import EvaluatorResult


class QATestCase(State, serializable=True):
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

Case selection follows the argument type: `None` (or no argument) runs every stored case, an `int`
runs that many random cases, a `float` in `(0, 1]` runs that fraction, and a sequence selects
specific cases by identifier, by parameters, or by `EvaluatorSuiteCase` instances. An unknown
identifier raises `ValueError`.

`storage` accepts a `Path`/`str` JSON file, a sequence of `EvaluatorSuiteCase` values for in-memory
cases, or a custom `EvaluatorSuiteCasesStorage` implementation; omitting it starts with empty
in-memory storage. File storage expects the file to exist - create it (`[]` for no cases) before the
first run. `with_storage(...)`, `with_name(...)`, `with_meta(...)`, `with_state(...)` and
`with_concurrent_evaluations(...)` produce reconfigured copies of a suite, and `prepared(...)` binds
extra arguments of the definition.

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
mean_eval = Evaluator.average(
    evaluator1.prepared(),
    evaluator2.prepared(),
    threshold="good",  # required - it replaces the combined thresholds
)
```

`lowest` and `highest` compare `performance` (score relative to each threshold) and return the
winning `EvaluatorResult` as is; `average` returns a new result named `average` carrying the mean
score. All three accept `concurrent_tasks` (2 by default).

```python
# Transform inputs before delegation
field_evaluator = my_evaluator.contra_map(MyModel._.attribute)
normalized = my_evaluator.contra_map(lambda data: data["response"].strip().lower())
```

## Reference-Based Scoring

When you have ground-truth ratings, score the evaluator itself: `Evaluator.referenced(...)` runs the
wrapped evaluation and replaces its score with how well that score conforms to an accepted window
pulled from the evaluated value. This keeps the ground truth in the value (e.g. a field of the suite
case parameters) instead of adding another argument.

```python
from draive import State
from draive.evaluation import EvaluationReference, evaluator


class ReviewCase(State, serializable=True):
    content: str
    expected: EvaluationReference


@evaluator(name="review_quality", threshold="good")
async def review_quality(case: ReviewCase) -> float:
    return await score_content(case.content)


# resolve the window per value - callable, attribute path, or a constant reference
referenced = review_quality.referenced(reference=ReviewCase._.expected)
nominal = review_quality.referenced(
    reference=EvaluationReference(lower=0.6, upper=1.0),
    weighting="nominal",
)
```

`EvaluationReference` describes the accepted window:

- `EvaluationReference.of("good")` - exact single point (0.6).
- `EvaluationReference.of("good", tolerance=0.2)` - `±20%` of the target, so `[0.48, 0.72]`.
- `EvaluationReference(lower=0.6, upper=1.0)` - explicit bounds.
- A bare score value (`"good"`, `0.6`, `True`) is accepted anywhere a reference is expected and
    treated as an exact single-point window.

`weighting` controls the falloff outside the window: `"quadratic"` (default) grants partial credit
that decays with the squared distance to the nearest bound, `"nominal"` scores a miss as `0.0`. The
result keeps the original evaluator name and threshold and records `predicted_score`,
`predicted_level`, `reference_lower`, `reference_upper`, `within_reference` and `reference_weighting`
in its metadata. `reference_conformance(score, reference, weighting=...)` exposes the same math
directly.

## Rater Agreement

To compare an automatic evaluator against human labels, use Cohen's kappa. The functional variants
work on level labels, while `cohen_kappa_evaluator` accepts score values, `EvaluationScore` or
`EvaluatorResult` instances on both sides and bins them into levels.

```python
from draive.evaluation import cohen_kappa, quadratic_weighted_kappa
from draive.evaluators import cohen_kappa_evaluator

nominal = cohen_kappa(["good", "excellent"], ["good", "good"])
ordinal = quadratic_weighted_kappa(["good", "excellent"], ["good", "good"])

agreement = await cohen_kappa_evaluator(
    automatic_results,  # Sequence of scores, EvaluationScore or EvaluatorResult
    reference=human_labels,
    weighting="quadratic",  # or "nominal"
)
```

The evaluator score is the selected kappa clamped to `[0, 1]`, with both variants plus
`exact_agreement`, `sample_count` and `weighting` reported in metadata. The `tools/evals/` suite in
the repository uses exactly this evaluator to verify the shipped evaluators against labeled
baselines.

## Evaluating Conversation Context

Every built-in content evaluator has a `*_context_evaluator` twin operating on a `ModelContext` - the
sequence of `ModelInput`/`ModelOutput` elements produced by a conversation - so multi-turn behaviour
can be judged as a whole. Their reference-style arguments are optional: when omitted the evaluator
judges the outputs on their own (for example internal consistency instead of consistency with a
reference). `tool_usage_context_evaluator` is context-only and checks the tool calls recorded in the
timeline.

## Stateful Evaluation with Haiway

```python
from haiway import State, ctx

from draive.evaluation import evaluator


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

from draive.evaluation import EvaluationScore, evaluator


@evaluator
async def evaluate_with_context(response: str) -> EvaluationScore:
    score, issues = await analyze_response(response)

    return EvaluationScore.of(
        score,
        meta={
            "timestamp": datetime.now().isoformat(),
            "issues_found": issues,
            "evaluation_model": "gpt-5.5",
            "confidence": 0.85,
        },
    )
```

## Generating Test Cases

`generate_cases` asks a model to synthesize new case parameters of the suite's own parameters type.
All arguments are keyword-only; `examples` defaults to the currently available cases and `persist`
controls whether the extended set is written back to storage.

```python
examples = [
    QATestCase(
        question="What is machine learning?",
        expected_topics=["algorithms", "data", "training"],
        min_length=150,
    ),
    QATestCase(
        question="Explain overfitting",
        expected_topics=["generalization", "validation"],
    ),
]

cases = await qa_test_suite.generate_cases(
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

Generation runs a model call per case, so it has to happen inside a scope providing a model. The
returned sequence contains only the newly generated cases; they are appended to the suite's cases
either way.

## Summary

- Flexible scoring with normalized values and named levels
- Composable evaluators with thresholds and metadata
- Scenario grouping for related checks
- Suite management with persistent storage and generation tools
- Reporting helpers for insight into failures and regressions
- Reference windows and Cohen's kappa for validating evaluators against ground truth
- Concurrent execution to balance latency and throughput

Use evaluators for quick checks, scenarios for logical groupings, and suites for comprehensive
regression coverage backed by persistent cases and automated generation.
