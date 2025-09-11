# Basic Evaluation Usage


```python
from draive.evaluation import evaluator, EvaluationScore
from draive import Multimodal

@evaluator(name="keyword_presence", threshold=0.8)
async def keyword_evaluator(
    content: Multimodal,
    /,
    required_keywords: list[str],
) -> EvaluationScore:
    text = str(content).lower()
    found_keywords = sum(1 for keyword in required_keywords if keyword.lower() in text)

    if not required_keywords:
        return EvaluationScore.of(
            0,
            comment="No keywords provided for evaluation",
        )

    score = found_keywords / len(required_keywords)
    return EvaluationScore.of(
        score,
        comment=f"Found {found_keywords}/{len(required_keywords)} required keywords",
    )
```

Using this evaluator is straightforward:

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

    print(f"Score: {result.score.value}")
    print(f"Passed: {result.passed}")

from draive.evaluators import (
    groundedness_evaluator,
    readability_evaluator,
    coherence_evaluator,
    coverage_evaluator
)

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

# Evaluate text readability
complex_text = """
The utilization of sophisticated methodological approaches in the
implementation of artificial intelligence systems necessitates comprehensive
understanding of underlying algorithmic paradigms.
"""

readability_result = await readability_evaluator(complex_text)
print(f"Readability: {readability_result.score.value}")

# Check content coherence
coherence_result = await coherence_evaluator(
    generated_text,
    reference=reference_text
)
print(f"Coherence: {coherence_result.score.value}")

# Verify coverage of key points
coverage_result = await coverage_evaluator(
    generated_text,
    reference=reference_text
)

from draive.evaluators import (
    helpfulness_evaluator,
    completeness_evaluator,
    safety_evaluator
)

user_query = "How do I reset my password?"
response = """
To reset your password:
1. Go to the login page
2. Click 'Forgot Password'
3. Enter your email address
4. Check your email for reset instructions
"""

# Evaluate how helpful the response is
helpfulness_result = await helpfulness_evaluator.with_threshold("excellent")(
    response,
    user_query=user_query
)
print(f"Helpfulness: {helpfulness_result.score.value} ({'✓' if helpfulness_result.passed else '✗'})")

# Check if response fully addresses the query
completeness_result = await completeness_evaluator.with_threshold("good")(
    response,
    user_query=user_query
)
print(f"Completeness: {completeness_result.score.value} ({'✓' if completeness_result.passed else '✗'})")

# Verify content safety - critical threshold
safety_result = await safety_evaluator.with_threshold("perfect")(response)

from draive.evaluators import (
    factual_accuracy_evaluator,
    creativity_evaluator,
    tone_style_evaluator
)

content = "Python is a programming language created by Guido van Rossum in 1991."

# Check factual accuracy (no reference needed) - high threshold for accuracy
accuracy_result = await factual_accuracy_evaluator.with_threshold("excellent")(content)
print(f"Factual Accuracy: {accuracy_result.score.value} ({'✓' if accuracy_result.passed else '✗'})")

# Evaluate creativity - moderate threshold for creative content
creative_content = "Learning Python is like learning to speak with computers - you start with simple words and gradually build conversations."

creativity_result = await creativity_evaluator.with_threshold("good")(creative_content)
print(f"Creativity: {creativity_result.score.value} ({'✓' if creativity_result.passed else '✗'})")

# Check tone alignment - high threshold for brand consistency
expected_tone = "Professional, educational, and accessible to beginners"

tone_result = await tone_style_evaluator.with_threshold("excellent")(
    creative_content,
    expected_tone_style=expected_tone
)

from draive.evaluators import (
    required_keywords_evaluator,
    forbidden_keywords_evaluator,
    similarity_evaluator
)

content = "Our AI-powered solution uses machine learning for data analysis"

# Check for required keywords - moderate threshold for keyword matching
keywords_result = await required_keywords_evaluator.with_threshold("good")(
    content,
    keywords=["AI", "machine learning", "data"],
    require_all=True
)
print(f"Required keywords present: {keywords_result.score.value} ({'✓' if keywords_result.passed else '✗'})")

# Check for forbidden content - perfect threshold for safety compliance
forbidden_result = await forbidden_keywords_evaluator.with_threshold("perfect")(
    content,
    keywords=["hack", "exploit", "unauthorized"]
)
print(f"Forbidden keywords absent: {forbidden_result.score.value} ({'✓' if forbidden_result.passed else '✗'})")

# Compare semantic similarity - moderate threshold for similarity matching
reference_content = "AI systems utilize ML algorithms for data processing"
similarity_result = await similarity_evaluator.with_threshold("good")(
    content,
    reference=reference_content
)
print(f"Similarity: {similarity_result.score.value} ({'✓' if similarity_result.passed else '✗'})")
```


```python
from collections.abc import Sequence
from draive.evaluation import evaluate, evaluator_scenario, EvaluatorResult
from draive.evaluators import (
    conciseness_evaluator,
    readability_evaluator,
    safety_evaluator,
    factual_accuracy_evaluator
)

@evaluator_scenario(name="comprehensive_content_quality")
async def content_quality_scenario(
    content: str,
    /,
    *,
    reference: str,
) -> Sequence[EvaluatorResult]:
    # Evaluate using multiple criteria with appropriate thresholds
    return await evaluate(
        content,
        conciseness_evaluator.with_threshold("excellent").prepared(reference=reference),
        readability_evaluator.with_threshold("good").prepared(),
        safety_evaluator.with_threshold("perfect").prepared(),  # Safety is critical
        factual_accuracy_evaluator.with_threshold("excellent").prepared(),
    )

# Use the scenario
evaluation_results = await content_quality_scenario(
    generated_text,
    reference=reference_text,
)

# Process results
all_passed = all(result.passed for result in evaluation_results)
avg_performance = sum(result.performance for result in evaluation_results) / len(evaluation_results)

print(f"All evaluations passed: {all_passed}")
print(f"Average performance: {avg_performance:.2f}%")

for result in evaluation_results:


```python
from draive.evaluators import (
    helpfulness_evaluator,
    completeness_evaluator,
    tone_style_evaluator,
    safety_evaluator
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

# Example usage
user_query = "How can I improve my Python skills?"
response = "To improve your Python skills, start with online tutorials, practice coding daily, and work on projects."
expected_tone = "Friendly, encouraging, and actionable"

results = await user_response_scenario(response, user_query, expected_tone)
all_passed = all(result.passed for result in results)
print(f"User response quality passed: {all_passed}")

# Show individual results
for result in results:


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
    # Generate content based on test case parameters
    content: str = await TextGeneration.generate(
        instructions=f"Write informative content about {parameters.topic}",
        input=parameters.reference_material,
    )
    return await evaluate(
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
    EvaluatorSuiteCase(
        parameters=ContentTestCase(
            topic="climate change",
            required_keywords=["temperature", "emissions", "global"],
            reference_material="Global temperatures have risen 1.1°C since pre-industrial times",
        )
    ),
    EvaluatorSuiteCase(
        parameters=ContentTestCase(
            topic="renewable energy",
            required_keywords=["solar", "sustainable", "energy"],
            reference_material="Solar and wind power are leading renewable energy sources",
        )
    ),
]

# Prepare suite with in-memory test cases storage
suite = content_generation_suite.with_storage(test_cases)

# Execute suite evaluation
suite_results = await suite()

print(f"Suite passed: {suite_results.passed}")
print(f"Cases passed: {sum(1 for case in suite_results.results if case.passed)}/{len(suite_results.results)}")

for case_result in suite_results.results:
    print(f"\nCase {case_result.case_identifier}:")
    print(f"  Passed: {case_result.passed}")


```python
# Create evaluator with custom metadata
custom_evaluator = keyword_evaluator.with_meta({
    "version": "1.0",
    "author": "evaluation_team",
})

# Combine evaluators using logical operations
from draive.evaluation import Evaluator

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


```python
from draive.evaluation import evaluate
from draive.evaluators import (
    safety_evaluator,
    readability_evaluator,
    factual_accuracy_evaluator
)

# Run multiple evaluators concurrently on the same content
content = "Your content to evaluate..."

results = await evaluate(
    content,
    safety_evaluator.with_threshold("perfect").prepared(),
    readability_evaluator.with_threshold("good").prepared(),
    factual_accuracy_evaluator.with_threshold("excellent").prepared(),
    concurrent_tasks=2  # Control concurrency level
)

# Process results
for result in results:
    print(f"{result.evaluator}: {result.score.value} ({'✓' if result.passed else '✗'})")

# Check if all evaluations passed
all_passed = all(result.passed for result in results)
print(f"All evaluations passed: {all_passed}")
```
