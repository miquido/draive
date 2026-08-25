# Basic Evaluation Guide

Use evaluations to automatically score and validate the outputs of your generative pipelines. This
guide walks through the core building blocks, shows how to combine Draive's built-in evaluators, and
highlights practical patterns for running repeatable quality checks.

## Prerequisites

- Python 3.14+ with Draive installed and your project configured to use the shared Haiway context
    (`ctx`).
- Provider credentials available through `load_env()` or your preferred secrets loader.
- Familiarity with async/await. All evaluation APIs are asynchronous.

> Tip: When experimenting interactively you can rely on `print(...)`. In production code prefer
> `ctx.log_info(...)`, `ctx.log_warning(...)`, etc., to integrate with Haiway observability.

## 1. Write Your First Evaluator

Evaluators are async callables decorated with `@evaluator`. They receive the content you want to
check, optional parameters, and return a raw score (0.0–1.0). The decorator then wraps that score
into an `EvaluatorResult` so you always get pass/fail metadata alongside the score.

```python
from collections.abc import Sequence

from draive.evaluation import evaluator, EvaluationScore
from draive import Multimodal, MultimodalContent

@evaluator(name="keyword_presence", threshold=0.8)
async def keyword_evaluator(
    content: Multimodal,
    /,
    *,
    required_keywords: Sequence[str],
) -> EvaluationScore:
    text = MultimodalContent.of(content).to_str().lower()
    if not required_keywords:
        return EvaluationScore.of(0.0, meta={"comment": "No keywords provided"})

    found = sum(1 for keyword in required_keywords if keyword.lower() in text)
    score = found / len(required_keywords)

    # `@evaluator` wraps this score into an EvaluatorResult with the active threshold
    return EvaluationScore.of(
        score,
        meta={"comment": f"Matched {found}/{len(required_keywords)} required keywords"},
    )
```

Key ideas:

- `name` identifies the evaluator in reports; it has to follow snake case rules.
- `threshold` defines the default pass/fail cutoff (`1` when not specified). You can override it
    later with `.with_threshold(...)`.
- Returning an `EvaluationScore` lets you attach metadata; a plain score value (`float`, `bool`, or
    a level name like `"good"`) is accepted as well and gets wrapped automatically.
- Everything human-readable travels in `meta` - `EvaluationScore` has no dedicated `comment` field,
    so use the `"comment"` metadata key as the built-in evaluators do.

## 2. Run an Evaluator Inside a Context Scope

All provider calls must run inside a Haiway context. Prepare the scope, generate or collect the
content to evaluate, and await your evaluator.

```python
from draive import ctx, load_env
from draive.evaluation import EvaluatorResult
from draive.openai import OpenAI, OpenAIResponsesConfig

load_env()

async with ctx.scope(
    "evaluation_example",
    OpenAIResponsesConfig(model="gpt-5.5"),
    disposables=(OpenAI(),),
):
    content = "AI and machine learning are transforming technology"

    result: EvaluatorResult = await keyword_evaluator(
        content,
        required_keywords=["AI", "machine learning", "technology"],
    )

    print(f"Score: {result.score:.2f}")
    print(f"Passed default threshold: {result.passed}")
    print(f"Comment: {result.meta.get('comment')}")
```

Each call returns an `EvaluatorResult`; use `.passed` to compare the score with the active threshold,
`.performance` for the score expressed as a percentage of the threshold, and `.meta["comment"]` (when
set by the evaluator) for human-readable feedback. `EvaluatorResult.score` is a plain `float`.

## 3. Explore Built-in Evaluators

Draive ships ready-to-use evaluators that cover most quality axes. Import them from
`draive.evaluators` and configure per use case. See
[Evaluator Catalog](EvaluatorCatalog.md) for the full list with parameters.

**Quality and Structure**

- `readability_evaluator` – rates how easily the content can be understood.
- `fluency_evaluator` – rates grammar, spelling, punctuation, and sentence structure.
- `coherence_evaluator` – rates structure and organization against a `reference`.
- `coverage_evaluator` – verifies whether the output covers the `reference` key points.
- `conciseness_evaluator` – rates brevity while still covering the `reference` key information.

**Trust and Safety**

- `safety_evaluator` – screens for harmful, dangerous, or inappropriate content.
- `jailbreak_evaluator` – rates how far the content stays clear of guardrail bypass attempts.
- `factual_accuracy_evaluator` / `truthfulness_evaluator` – check factual correctness, optionally
    against a `reference`.
- `groundedness_evaluator` – ensures outputs map to supporting `reference` material.

**Interaction Quality**

- `helpfulness_evaluator`, `completeness_evaluator` – score responses against a `user_query`.
- `tone_style_evaluator` – scores content against an `expected_tone_style`.
- `expectations_evaluator` – scores content against explicit `expectations`.
- `required_keywords_evaluator` / `forbidden_keywords_evaluator` – enforce terminology.
- `similarity_evaluator` – compares semantic similarity to a `reference`.

Every content evaluator listed above also has a `*_context_evaluator` twin (for example
`safety_context_evaluator`) that takes a whole `ModelContext` - the conversation timeline of
`ModelInput`/`ModelOutput` elements - instead of a single piece of content.

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
    print(f"{label}: {result.score:.2f} ({'✓' if result.passed else '✗'})")
```

Adjust thresholds by chaining `.with_threshold("good")`, `.with_threshold("excellent")`, etc. All
evaluators share one level scale: `"none"` (0.0), `"poor"` (0.2), `"fair"` (0.4), `"good"` (0.6),
`"excellent"` (0.8), `"perfect"` (1.0). Raw floats and booleans work as thresholds too.

## 4. Combine Evaluators with Scenarios

Use `@evaluator_scenario` to bundle several evaluators into a reusable checklist. The scenario
definition returns a sequence of `EvaluatorResult` objects, while calling the scenario yields a
single `EvaluatorScenarioResult` holding them - so you can compute aggregates or present detailed
feedback.

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
scenario_result = await user_response_scenario(
    response,
    user_query=user_query,
    expected_tone=expected_tone,
)

print(f"All checks passed: {scenario_result.passed}")
for item in scenario_result.results:
    print(f"- {item.evaluator}: {item.score:.2f} ({'✓' if item.passed else '✗'})")

# or let the result render itself
print(scenario_result.report(detailed=False))
```

## 5. Automate Regression Checks with Suites

Suites orchestrate content generation and evaluation over structured test cases. Use them for
nightly quality gates or pre-release validation.

```python
from collections.abc import Sequence
from draive.evaluation import evaluator_suite, evaluate, EvaluatorResult, EvaluatorSuiteCase
from draive import State, TextGeneration

class ContentTestCase(State, serializable=True):
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

Each case produces an `EvaluatorSuiteCaseResult` (with `.case_identifier`, `.results`, `.passed`,
`.performance` and `.report(...)`). You can persist these to dashboards, CI artifacts, or team
reports. Cases without an explicit `identifier` get a generated UUID.

`with_storage(...)` (or the `storage=` argument of `@evaluator_suite`) accepts an in-memory sequence
of cases, a `Path`/`str` pointing at a JSON file, or any custom `EvaluatorSuiteCasesStorage`. Calling
the suite with no argument runs every stored case; pass an `int` for that many random cases, a
`float` between 0 and 1 for a fraction of them, or a sequence of identifiers, parameters, or cases to
run a specific selection.

## 6. Advanced Patterns

- **Attach metadata**: `keyword_evaluator.with_meta({"version": "1.0"})` adds context that surfaces
    in result payloads.
- **Compose evaluators**: `Evaluator.highest(...)`, `Evaluator.lowest(...)` and
    `Evaluator.average(..., threshold=...)` combine prepared evaluators into a single one keeping the
    best, worst, or mean outcome.
- **Adapt inputs**: `.contra_map(lambda doc: doc.body)` transforms incoming data before evaluation,
    perfect for domain models.
- **Score against ground truth**: `.referenced(reference=...)` replaces the produced score with how
    well it agrees with an expected score window - see
    [Comprehensive Evaluation](ComprehensiveEvaluation.md).
- **Inject state**: `@evaluator(state=(MyConfig(),))` or `.with_state(MyConfig())` makes state
    available within the evaluator scope.
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

- Continue with [Comprehensive Evaluation](ComprehensiveEvaluation.md) for suites, reference-based
    scoring, and rater agreement.
- Browse the [Evaluator Catalog](EvaluatorCatalog.md) for every shipped evaluator and its parameters,
    or read the sources under `src/draive/evaluators/` for inspiration.
- Extend scenarios with custom analytics by post-processing `EvaluatorResult.performance` across
    runs.

With these building blocks you can turn qualitative reviews into automated guardrails that keep your
agents and workflows on target.
