# Basic Decision Generation

Use `DecisionGeneration` to make typed decisions about content: yes / no checks, single-select
choices, multi-select selections and ratings on ordered scales.

Decisions follow the contract of dedicated decision models - one shared input is evaluated against a
set of independent, named questions. A backend built for decisions, such as the Gemini Decisions
API, Ollama decision models or any System One API server, additionally reports calibrated
probabilities you can threshold. A decision backend always has to be bound, either one of those or
the optional completion based implementation described below.

`DecisionGeneration` is the high level facade (templates, `check`, `choose`, `select`, `rate`). It
sits on `DecisionModel` from `draive.models`, the thin backend layer next to `GenerativeModel` which
providers install and which `Step.route` uses directly. Without an explicit `DecisionGeneration`,
the facade uses the `DecisionModel` bound in the scope.

Reach for decisions whenever the answer is one of a fixed set of outcomes rather than generated
content: classifying or routing a ticket, moderating input, tagging topics, grading a draft against
a rubric, or gating a workflow step. Compared to asking `ModelGeneration` for an enum field, every
question is answered independently, decided options come back as your own `DecisionOption`s with
their meta, and dedicated backends tell you how certain each decision is, so uncertain cases can be
escalated instead of trusted blindly.

## Prerequisites

- Python 3.15+
- Gemini (`GEMINI_API_KEY`) or System One API (`SYSTEMONE_API_KEY`) credentials loaded via
    `load_env()`, or a local Ollama server

```python
from draive import load_env

load_env()
```

## 1. Ask Questions

Three question kinds are available, each answered with its own answer type:

| Question            | Answer                    | Decided value                                             |
| ------------------- | ------------------------- | --------------------------------------------------------- |
| `DecisionCheck`     | `DecisionCheckAnswer`     | `value: bool`                                             |
| `DecisionChoice`    | `DecisionChoiceAnswer`    | `choice: DecisionOption` - one of the options             |
| `DecisionSelection` | `DecisionSelectionAnswer` | `selected: Sequence[DecisionOption]` - any of the options |
| `DecisionRating`    | `DecisionRatingAnswer`    | `level: DecisionOption` - one of the levels               |

Options and rating levels are all described with `DecisionOption`, an `identifier` with an optional
`description`. Answers carry the whole decided option, the decided value is all every backend
guarantees - anything a particular backend reports beyond it lands in the answer `meta`.

`decide` answers any number of questions about the same input at once, each question is evaluated
independently and can't see the others. Every question carries a unique `identifier`, answers are
returned in the order of their questions and carry the `identifier` of the question they answer.

```python
from draive import (
    DecisionCheck,
    DecisionChoice,
    DecisionGeneration,
    DecisionOption,
    DecisionRating,
)

angry, route, urgency = await DecisionGeneration.decide(
    input="Customer message: I was charged twice and nobody has replied.",
    questions=(
        DecisionCheck(identifier="angry", criteria="Is the customer angry?"),
        DecisionChoice(
            identifier="route",
            criteria="Which team should handle this?",
            # description is optional, skip it when the identifier is self-explanatory
            options=(
                DecisionOption(identifier="billing", description="payment issues"),
                DecisionOption(identifier="shipping"),
                DecisionOption(identifier="technical"),
            ),
        ),
        DecisionRating(
            identifier="urgency",
            criteria="How urgent is this?",
            # ordered from the lowest to the highest level
            levels=(
                DecisionOption(identifier="can wait"),
                DecisionOption(identifier="this week"),
                DecisionOption(identifier="today", description="before the end of the day"),
                DecisionOption(identifier="right now"),
            ),
        ),
    ),
)
route.identifier  # "route"
route.choice.identifier  # "billing"
```

Questions are validated when created: criteria and identifiers can't be empty, a choice needs at
least two options and a rating at least two levels, each with a unique identifier. Malformed
questions raise `ValidationError`.

For a single question use the typed helpers instead, they return the matching answer type directly:

```python
route = await DecisionGeneration.choose(
    input=ticket,
    criteria="Which team should handle this?",
    options=(
        DecisionOption(identifier="billing", description="payment issues"),
        DecisionOption(identifier="shipping"),
    ),
)
damaged = await DecisionGeneration.check(
    input=photo,
    criteria="Did the item arrive damaged?",
)
quality = await DecisionGeneration.rate(
    input=draft,
    criteria="How well does the draft answer the question?",
    levels=(
        DecisionOption(identifier="off topic"),
        DecisionOption(identifier="partial"),
        DecisionOption(identifier="complete"),
    ),
)
topics = await DecisionGeneration.select(
    input=ticket,
    criteria="Which topics does the ticket mention?",
    options=(
        DecisionOption(identifier="billing"),
        DecisionOption(identifier="shipping"),
        DecisionOption(identifier="refund"),
    ),
)
```

Each option of a selection is judged independently of the others, so any number of them, including
none, can be selected. Selected options are returned in the order of the question options.

Input can be any multimodal content or a `Template`.

Options can carry your own values in `meta`, which are never sent to the backend. The decided
option's meta is merged into the meta of choice and rating answers, so you can act on a decision
without mapping identifiers back to your own values. Selected options carry their own meta, it is
not merged as values of several options could collide:

```python
route = await DecisionGeneration.choose(
    input=ticket,
    criteria="Which team should handle this?",
    options=(
        DecisionOption(identifier="billing", meta={"queue": "billing-l1"}),
        DecisionOption(identifier="shipping", meta={"queue": "logistics"}),
    ),
)
route.meta["queue"]  # "billing-l1" when billing was decided
```

## 2. Use Backend Specific Values

Backends put what they report beyond the decision into the answer `meta`, merged over the decided
option's meta. Gemini and System One servers report calibrated estimates:

| Key             | Answers                   | Value                                                    |
| --------------- | ------------------------- | -------------------------------------------------------- |
| `probability`   | check                     | probability of the positive outcome                      |
| `probabilities` | choice, selection, rating | probability of each option or level, keyed by identifier |
| `confidence`    | check, choice, rating     | probability of the decided outcome                       |
| `score`         | rating                    | expected position from 0 (lowest level) to 1 (highest)   |

Use them to route uncertain cases to a fallback instead of trusting every decision:

```python
confidence = route.meta.get_float("confidence")
if confidence is not None and confidence < 0.7:
    ...  # escalate to a human
```

The completion based implementation has no calibrated estimates, it reports the model `reasoning`
behind each decision instead. Thresholds therefore need a calibrated backend.

## 3. Bind Gemini Decisions

`Gemini` prepares `DecisionModel` by default, alongside its other features. Pass `features` without
it to skip it. `GeminiDecisionConfig` selects the model, it has to be set explicitly.

```python
from draive import DecisionGeneration, ctx
from draive.gemini import Gemini, GeminiDecisionConfig

async with ctx.scope(
    "decisions",
    GeminiDecisionConfig(model="gemini-decision-1"),
    disposables=(Gemini(),),
):
    answers = await DecisionGeneration.decide(input=ticket, questions=questions)
```

The Gemini Decisions API is not available through Vertex AI, a `Gemini()` client using Vertex AI
does not provide `DecisionModel` unless it is requested in `features` explicitly. Its limits are
validated before sending a request: up to 1000 questions, up to 255 options or levels, and choice
option identifiers up to 128 characters without control characters. The API has no multi-select
question, each selection option is asked as a separate yes / no question within the same request and
counts against the question limit. Supported input is text, images, audio, video and PDF documents.

## 4. Bind Ollama Decisions

`Ollama()` provides `DecisionModel` out of the box, answered by a System One decision model served
by Ollama (0.35 or newer). Decision models are separate from chat models and have to be pulled
first, i.e. `ollama pull tev1`. `OllamaDecisionConfig` selects the model, it has to be set
explicitly.

```python
from draive import DecisionGeneration, ctx
from draive.ollama import Ollama, OllamaChatConfig, OllamaDecisionConfig

async with ctx.scope(
    "decisions",
    OllamaChatConfig(model="llama3.2"),
    OllamaDecisionConfig(model="tev1"),
    disposables=(Ollama(),),
):
    answers = await DecisionGeneration.decide(input=ticket, questions=questions)
```

The Ollama client delivers text input only. To decide about images use `SystemOne` with
`images="images"` against the same server, as described below. Selections count against the Ollama
limit of 64 questions per request, each option as a separate question. Without a pulled decision
model, use `Ollama(features=(GenerativeModel, TextEmbedding))` to skip `DecisionModel` and bind the
completion based decisions instead.

## 5. Bind System One API Decisions

`SystemOne` speaks the System One API (`POST /v1/systemone`), introduced with TypeSafe Jev and
served by several backends: the hosted [TypeSafe API](https://docs.typesafe.ai/api), local
[Ollama](https://docs.ollama.com/api/systemone) (0.35 or newer, with a decision model such as `tev1`
or `nimble`), gateways such as OpenRouter, and self-hosted
[OpenJev](https://huggingface.co/openjev/openjev) servers. Install the `systemone` extra:

```bash
pip install 'draive[systemone]'
```

By default `SystemOne` targets TypeSafe - the server address resolves from `SYSTEMONE_SERVER_URL`
(falling back to `https://api.typesafe.ai`) and the bearer token from `SYSTEMONE_API_KEY` or
`TYPESAFE_API_KEY`. `SystemOneConfig` selects the model, `jev-latest` by default.

```python
from draive import DecisionGeneration, ctx
from draive.systemone import SystemOne, SystemOneConfig

async with ctx.scope(
    "decisions",
    SystemOneConfig(model="jev-latest"),
    disposables=(SystemOne(),),
):
    answers = await DecisionGeneration.decide(input=ticket, questions=questions)
```

Servers differ in how they accept images, select it with `images`:

- `"assets"` (default, TypeSafe) - any number of png / jpeg images, sent as request assets placed
    where they appear within the input. TypeSafe serves images only with a vision model such as
    `jev-preview`, the required `TypeSafe-Danger-Zone-Images` header is added automatically.
- `"images"` (Ollama) - images shared by all questions, sent as a separate list next to the textual
    state, which is still required. Images need a model with vision support.
- `"inline"` (OpenJev) - text with at most one image, sent within the state.

```python
from draive.systemone import SystemOne, SystemOneConfig

ollama = SystemOne(server_url="http://localhost:11434", images="images")
config = SystemOneConfig(model="tev1")
```

Other TypeSafe feature flags, such as `TypeSafe-Danger-Zone-Long-Context`, can be passed through
`headers`. Answer meta carries the same calibrated values as with Gemini, `confidence` is reported
as the probability of the decided outcome instead of the concentration measure servers report
themselves. Selections are resolved the same way as with Gemini, as a yes / no question per option -
mind the question limit of the server, i.e. 64 questions for Ollama.

## 6. Completion Based Decisions

`generate_completion_decisions` is an optional backend answering every question with a separate
structured `GenerativeModel` completion, without tools, using the model bound in the scope, up to 8
of them running concurrently. It is never used implicitly - completions may be slow and costly, so
it has to be assigned explicitly, to `DecisionGeneration` or to `DecisionModel` (for `Step.route`).
A selection is answered with a single completion listing every option which applies. Answers outside
of the given options or levels raise `GenerationOutputMissing`. The model reasoning behind each
decision is reported in the answer meta under `reasoning`. Questions are answered concurrently, so
several of them failing together raise an `ExceptionGroup` of the underlying failures -
`GenerationOutputMissing` for invalid answers, or model errors such as `ModelRateLimit`. Handle the
grouped case as well when calling `decide` with multiple questions.

```python
from draive import DecisionGeneration, DecisionOption, ctx, generate_completion_decisions
from draive.openai import OpenAI, OpenAIResponsesConfig

async with ctx.scope(
    "decisions",
    OpenAIResponsesConfig(model="gpt-5.5"),
    DecisionGeneration(generating=generate_completion_decisions),
    disposables=(OpenAI(),),
):
    route = await DecisionGeneration.choose(
        input=ticket,
        criteria="Which team should handle this?",
        options=(
            DecisionOption(identifier="billing"),
            DecisionOption(identifier="shipping"),
        ),
    )
```

## 7. Custom Backends

Bind `DecisionGeneration(generating=...)` with any function matching `DecisionGenerating` to plug in
another decision backend. `DecisionGeneration` passes the backend answers through unchanged, so the
backend is responsible for the answer contract: one answer for every question, in the order of the
questions, each carrying the identifier of its question. Choice and rating answers carry the decided
option of the question, with its meta merged into the answer meta under any backend specific values:

```python
from collections.abc import Sequence
from typing import Any

from draive import DecisionAnswer, DecisionGeneration, DecisionQuestion, Multimodal, ctx


async def decide(
    *,
    input: Multimodal,
    questions: Sequence[DecisionQuestion],
    **extra: Any,
) -> Sequence[DecisionAnswer]:
    # answer each question in order, setting the question identifier on its answer,
    # merging the decided option meta with backend specific values on choice and rating
    ...


async with ctx.scope("decisions", DecisionGeneration(generating=decide)):
    ...
```
