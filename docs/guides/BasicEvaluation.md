# Basic Evaluation Guide

Use evaluations to automatically score and validate the outputs of your generative pipelines. This
guide walks through the core building blocks, shows how to combine Draive's built-in evaluators, and
highlights practical patterns for running repeatable quality checks.

## Prerequisites

- Python 3.13+ with Draive installed and your project configured to use the shared Haiway context
  (`ctx`).
- Provider credentials available through `load_env()` or your preferred secrets loader.
- Familiarity with async/await. All evaluation APIs are asynchronous.

> Tip: When experimenting interactively you can rely on `print(...)`. In production code prefer
> `ctx.log_info(...)`, `ctx.log_warn(...)`, etc., to integrate with Haiway observability.

## 1. Write Your First Evaluator

Evaluators are async callables decorated with `@evaluator`. They receive the content you want to
check, optional parameters, and return an `EvaluationScore` with a numeric score (0.0–1.0) and
metadata about the decision.

```python
from draive.evaluation import evaluator, EvaluationScore
from draive import Multimodal

@evaluator(name="keyword_presence", threshold=0.8)
async def keyword_evaluator(
    content: Multimodal,
    /,
    *,
    required_keywords: list[str],
) -> EvaluationScore:
    text = str(content).lower()
    if not required_keywords:
        return EvaluationScore.of(0, comment="No keywords provided")

    found = sum(1 for keyword in required_keywords if keyword.lower() in text)
    score = found / len(required_keywords)

    return EvaluationScore.of(
        score,
        comment=f"Matched {found}/{len(required_keywords)} required keywords",
    )
```

Key ideas:

- `name` identifies the evaluator in reports.
- `threshold` defines the default pass/fail cutoff. You can override it later with
  `.with_threshold(...)`.
- Always return an `EvaluationScore` so downstream tooling has consistent metadata.

## 2. Run an Evaluator Inside a Context Scope

All provider calls must run inside a Haiway context. Prepare the scope, generate or collect the
content to evaluate, and await your evaluator.

```python
from draive import ctx, load_env
from draive.openai import OpenAI, OpenAIResponsesConfig

load_env()

async with ctx.scope(
    "evaluation_example",
    OpenAIResponsesConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
):
    content = "AI and machine learning are transforming technology"

    result = await keyword_evaluator(
        content,
        required_keywords=["AI", "machine learning", "technology"],
    )

    print(f"Score: {result.score.value:.2f}")
    print(f"Passed default threshold: {result.passed}")
```

`EvaluationScore.passed` compares the computed score with the evaluator's active threshold. Use
`.comment` for human-readable feedback when showing results to reviewers.

## 3. Explore Built-in Evaluators

Draive ships ready-to-use evaluators that cover most quality axes. Import them from
`draive.evaluators` and configure per use case.

**Quality and Structure**

- `readability_evaluator` – favors concise, accessible language.
- `coherence_evaluator` – checks internal consistency.
- `coverage_evaluator` – verifies whether the output covers reference points.
- `conciseness_evaluator` – penalizes overly long responses.

**Trust and Safety**

- `safety_evaluator` – screens for policy violations.
- `factual_accuracy_evaluator` – checks factual alignment.
- `groundedness_evaluator` – ensures outputs map to supporting references.

**Interaction Quality**

- `helpfulness_evaluator`, `completeness_evaluator`, `tone_style_evaluator` – score responses to
  user prompts.
- `required_keywords_evaluator` / `forbidden_keywords_evaluator` – enforce terminology.
- `similarity_evaluator` – compares semantic similarity to a reference.

### Example: Stack Multiple Built-ins

```python
from draive.evaluators import (
    groundedness_evaluator,
    readability_evaluator,
    coherence_evaluator,
    coverage_evaluator,
)

reference_text = (
    "Climate change is causing rising sea levels globally.\n"
    "Scientific data shows ocean levels have risen 8-9 inches since 1880."
)

generated_text = (
    "Based on scientific evidence, global sea levels have increased\n"
    "approximately 8-9 inches since 1880 due to climate change impacts."
)

groundedness = await groundedness_evaluator(
    generated_text,
    reference=reference_text,
)
readability = await readability_evaluator(generated_text)
coherence = await coherence_evaluator(
    generated_text,
    reference=reference_text,
)
coverage = await coverage_evaluator(
    generated_text,
    reference=reference_text,
)

for label, result in {
    "Groundedness": groundedness,
    "Readability": readability,
    "Coherence": coherence,
    "Coverage": coverage,
}.items():
    print(f"{label}: {result.score.value:.2f} ({'✓' if result.passed else '✗'})")
```

Adjust thresholds by chaining `.with_threshold("good")`, `.with_threshold("excellent")`, etc. Each
evaluator documents its supported levels.

## 4. Combine Evaluators with Scenarios

Use `@evaluator_scenario` to bundle several evaluators into a reusable checklist. Scenarios return a
sequence of `EvaluatorResult` objects so you can compute aggregates or present detailed feedback.

```python
from collections.abc import Sequence
from draive.evaluation import evaluate, evaluator_scenario, EvaluatorResult
from draive.evaluators import (
    helpfulness_evaluator,
    completeness_evaluator,
    tone_style_evaluator,
    safety_evaluator,
)

@evaluator_scenario(name="user_response_quality")
async def user_response_scenario(
    response: str,
    user_query: str,
    expected_tone: str,
) -> Sequence[EvaluatorResult]:
    return await evaluate(
        response,
        helpfulness_evaluator.with_threshold("excellent").prepared(user_query=user_query),
        completeness_evaluator.with_threshold("good").prepared(user_query=user_query),
        tone_style_evaluator.with_threshold("good").prepared(expected_tone_style=expected_tone),
        safety_evaluator.with_threshold("perfect").prepared(),
    )
```

Run the scenario and inspect individual checks:

```python
results = await user_response_scenario(
    response,
    user_query=user_query,
    expected_tone=expected_tone,
)

all_passed = all(result.passed for result in results)
print(f"All checks passed: {all_passed}")
for item in results:
    print(f"- {item.evaluator}: {item.score.value:.2f} ({'✓' if item.passed else '✗'})")
```

## 5. Automate Regression Checks with Suites

Suites orchestrate content generation and evaluation over structured test cases. Use them for
nightly quality gates or pre-release validation.

```python
from typing import Sequence
from draive.evaluation import evaluator_suite, evaluate, EvaluatorResult, EvaluatorSuiteCase
from draive import TextGeneration, DataModel

class ContentTestCase(DataModel):
    topic: str
    required_keywords: Sequence[str]
    reference_material: str

@evaluator_suite(ContentTestCase)
async def content_generation_suite(
    parameters: ContentTestCase,
) -> Sequence[EvaluatorResult]:
    content = await TextGeneration.generate(
        instructions=f"Write informative content about {parameters.topic}",
        input=parameters.reference_material,
    )
    return await evaluate(
        content,
        keyword_evaluator.with_threshold(0.5).prepared(
            required_keywords=parameters.required_keywords,
        ),
        groundedness_evaluator.prepared(reference=parameters.reference_material),
        readability_evaluator.prepared(),
    )
```

Create cases and run the suite:

```python
test_cases = [
    EvaluatorSuiteCase(
        parameters=ContentTestCase(
            topic="climate change",
            required_keywords=["temperature", "emissions", "global"],
            reference_material="Global temperatures have risen 1.1°C since pre-industrial times",
        ),
    ),
    EvaluatorSuiteCase(
        parameters=ContentTestCase(
            topic="renewable energy",
            required_keywords=["solar", "sustainable", "energy"],
            reference_material="Solar and wind power are leading renewable energy sources",
        ),
    ),
]

suite = content_generation_suite.with_storage(test_cases)
suite_results = await suite()

print(f"Suite passed: {suite_results.passed}")
print(
    "Cases passed: "
    f"{sum(1 for case in suite_results.results if case.passed)}/{len(suite_results.results)}"
)
```

Each `EvaluatorSuiteCase` produces a detailed result object. You can persist these to dashboards, CI
artifacts, or team reports.

## 6. Advanced Patterns

- **Attach metadata**: `keyword_evaluator.with_meta({"version": "1.0"})` adds context that surfaces
  in result payloads.
- **Compose evaluators**: `Evaluator.highest(...)` and `Evaluator.lowest(...)` let you compare
  multiple evaluators and keep the best/worst outcome.
- **Adapt inputs**: `.contra_map(lambda doc: doc.body)` transforms incoming data before evaluation,
  perfect for domain models.
- **Control concurrency**: `evaluate(..., concurrent_tasks=2)` balances throughput with provider
  rate limits when running many checks at once.
- **Tune thresholds per run**: Choose qualitative targets (`"good"`, `"excellent"`, etc.) or numeric
  thresholds when converting results into pass/fail signals for CI.

## 7. Troubleshooting and Best Practices

- Start with generous thresholds to establish a baseline, then tighten as you collect data.
- Log both scores and comments so reviewers understand failures quickly.
- Use scenarios for deterministic evaluations and suites when content generation is part of the
  test.
- Mock provider calls in unit tests; evaluation functions themselves remain pure async callables.
- Keep evaluators small and single-purpose. Compose rather than creating monoliths.

## Next Steps

- Dive into the full API reference in `docs/reference/evaluation.md` (or run `make docs-server` to
  explore locally).
- Explore domain-specific evaluators under `draive/evaluators/` for inspiration.
- Extend scenarios with custom analytics by post-processing `EvaluatorResult.performance` across
  runs.

With these building blocks you can turn qualitative reviews into automated guardrails that keep your
agents and workflows on target.
