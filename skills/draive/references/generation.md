# Generation - providers, text/structured/image/audio, decisions, content, templates, tools, skills

Everything here is `from draive import ...` unless noted. Generation facades are Haiway `State`s
with `statemethod`s: call `TextGeneration.generate(...)` on the class and it resolves the instance
from the active scope (an absent one is built from its default, so no wiring is needed for
facades). The model comes from the scope - never pass a provider around.

## Scoping a provider

A provider is a disposable client that installs states when the scope opens; the config `State`
selects the model.

```python
from draive import ctx, load_env
from draive.openai import OpenAI, OpenAIResponsesConfig

load_env()                                       # OPENAI_API_KEY etc. from .env
async with ctx.scope(
    "app",
    OpenAIResponsesConfig(model="gpt-5.5", reasoning="none"),
    disposables=(OpenAI(),),                     # installs GenerativeModel, ImageGeneration, ...
):
    ...
```

| Provider | Disposable | Config state |
| --- | --- | --- |
| OpenAI | `draive.openai.OpenAI` | `OpenAIResponsesConfig` (`OpenAIImageGenerationConfig`, `OpenAIEmbeddingConfig`, ...) |
| Anthropic | `draive.anthropic.Anthropic` | `AnthropicConfig` |
| Gemini | `draive.gemini.Gemini` | `GeminiConfig` (`GeminiDecisionConfig`, ...) |
| Mistral / Ollama | `draive.mistral.Mistral` / `draive.ollama.Ollama` | `MistralChatConfig` / `OllamaChatConfig` (`OllamaDecisionConfig`) |
| System One (decisions only) | `draive.systemone.SystemOne` | `SystemOneConfig` |

- Each client takes `features=` - the set of state types it installs - and reads credentials from
  the environment (`api_key=` overrides, where the provider has one). `Anthropic()` installs only
  `GenerativeModel`; `OpenAI()` also installs realtime, embedding, image generation and moderation;
  `Gemini()` and `Ollama()` also install `DecisionGeneration`. The full table is in
  `providers-and-data.md`.
- Change config for a block: `with
  ctx.updating(ctx.state(OpenAIResponsesConfig).updating(model=...)):`. For one call pass `config=`
  - it travels through `**extra` to the provider: `TextGeneration.generate(...,
  config=ctx.state(OpenAIResponsesConfig).updating(reasoning="low"))`.
- Provider extras are optional installs (`draive[openai]`); see `providers-and-data.md`.

## Text, structured, image, audio

```python
from draive import ImageGeneration, ModelGeneration, State, TextGeneration, tool

class City(State, serializable=True):
    name: str
    population: int | None = None            # optional when "absent" is a valid answer

text: str = await TextGeneration.generate(
    instructions="Be concise.",              # str, or a Template
    input="Hello",                           # Multimodal (str, content, parts), or a Template
    tools=[current_time],                    # Toolbox | Iterable[Tool]; tool loop is automatic
    examples=(("2+2", "4"),),                # few-shot (input, output) pairs
)
city: City = await ModelGeneration.generate(
    City,                                    # positional, a serializable State type
    instructions="Extract the city.",
    input="Oslo has 700k people.",
)
image = await ImageGeneration.generate(input="a red cat")    # ResourceContent | ResourceReference
```

- These four share `instructions`, `input`, `**extra`; keyword-only. `AudioGeneration` is
  the same as `ImageGeneration`. Text and model also take `tools=` and `examples=`
  (`(Multimodal, Generated)` pairs for `ModelGeneration`).
- `ModelGeneration` always requests `generated` as the output schema from the provider; extra
  options: `schema_injection="skip" | "simplified" | "full"` renders the schema into the
  instructions where a `{model_schema}` placeholder stands (without the placeholder nothing
  changes; in a `Template` use the template syntax `{%model_schema%}`), and
  `decoder=fn(MultimodalContent) -> Generated` to read the result yourself. Default decoding takes
  a JSON artifact if present, else parses the text as JSON.
- Strict-schema providers fill every field, so a default is not "left out" - model absence as
  `X | None`; `Any`, `tuple[...]` and recursive fields lose strictness on OpenAI.
- Image/audio raise `GenerationOutputMissing` when the model returned no such output; the default
  goes through `GenerativeModel` with image/audio output, `OpenAI()` installs its own image one.
- Pipelines of calls are `Step`s and agents; see `workflows.md`.

## Decisions

`DecisionGeneration` answers typed questions about one shared input - classification, routing,
moderation, rubric scoring - instead of generating content. Dedicated decision models (Gemini
Decisions API, System One models through `SystemOne` or `Ollama()`) answer with calibrated
probabilities you can threshold; without one bound, each question becomes a structured
`GenerativeModel` completion of the scope model (no probabilities, `reasoning` in meta).

```python
from draive import (
    DecisionCheck, DecisionChoice, DecisionGeneration, DecisionOption, DecisionRating,
    DecisionSelection,
)

billing = DecisionOption(identifier="billing", description="payment issues", meta={"queue": "l1"})
shipping = DecisionOption(identifier="shipping", meta={"queue": "logistics"})
angry, route, topics, urgency = await DecisionGeneration.decide(
    input=ticket,                            # Multimodal or a Template, shared by all questions
    questions=(                              # unique identifiers, answered independently
        DecisionCheck(identifier="angry", criteria="Is the customer angry?"),
        DecisionChoice(identifier="route", criteria="Which team?",
                       options=(billing, shipping)),
        DecisionSelection(identifier="topics", criteria="Which topics apply?",
                          options=(billing, DecisionOption(identifier="delivery"))),
        DecisionRating(identifier="urgency", criteria="How urgent?",       # lowest to highest
                       levels=(DecisionOption(identifier="low"), DecisionOption(identifier="high"))),
    ),
)
route.choice.identifier, route.meta["queue"]      # decided option, its meta merged into the answer
confidence = route.meta.get_float("confidence")   # calibrated only with a decision backend bound
answer = await DecisionGeneration.choose(input=ticket, criteria="Which team?", options=(...))
```

- Question kinds: `DecisionCheck` (yes / no -> `value: bool`), `DecisionChoice` (exactly one of
  2+ options -> `choice`), `DecisionSelection` (any of 1+ options, possibly none -> `selected`),
  `DecisionRating` (one of 2+ ordered levels -> `level`). Shortcuts for one question: `check`,
  `choose`, `select`, `rate` return the typed answer.
- Questions take `criteria` (not `instructions`); options and levels are `DecisionOption`
  (`identifier`, optional `description`, `meta` never sent to the backend). Invalid questions
  (empty criteria, duplicated option identifiers, too few options) fail at construction with
  `ValidationError`.
- `decide` returns answers in question order, each carrying its question `identifier`. The decided
  value is all every backend guarantees; everything else lives in answer `meta`: calibrated
  `probability` (check), `probabilities` keyed by option identifier, `confidence` (probability of
  the decided outcome), normalized 0-1 `score` (rating), or `reasoning` from the model fallback.
  Choice and rating answers merge the decided option meta in, selected options keep their own.
- Prefer it over `ModelGeneration` with an enum field when the job is picking among fixed options,
  especially when thresholds matter - but only threshold with a calibrated backend bound: the model
  fallback reports no estimates, and option meta merged into the answer could carry a
  `confidence` key of its own, so know which backend the scope binds.
- The model fallback answers questions concurrently: several failing together raise an
  `ExceptionGroup` of the underlying failures (`GenerationOutputMissing` for invalid answers, model
  errors such as `ModelRateLimit`), a single one raises directly.
- Bind a backend like any provider: `Gemini()` (`GeminiDecisionConfig`, model set explicitly, on by default), `Ollama()`
  (`OllamaDecisionConfig`, model set explicitly, needs a pulled decision model such as `tev1`, text only) or
  `SystemOne(server_url=..., images=...)` for TypeSafe Jev, Ollama with images, gateways.
  `Gemini(features=...)` / `Ollama(features=...)` without `DecisionGeneration` keeps the model
  fallback. Guide: `docs/guides/BasicDecisionGeneration.md`.

## Content: `MultimodalContent`

An immutable container of parts: `TextContent`, `ResourceContent` (inline bytes),
`ResourceReference` (URI), `ArtifactContent` (typed `State` payload), plus `MultimodalTag`.

```python
from draive import ArtifactContent, MultimodalContent, ResourceContent, State, TextContent

class Summary(State, serializable=True):
    title: str

content = MultimodalContent.of(
    "Describe this:",                                    # str -> TextContent
    ResourceContent.of(png_bytes, mime_type="image/png"),
    ArtifactContent.of(Summary(title="t"), category="summary"),
)
content.to_str()                       # text; resources as markdown placeholders, artifacts as JSON
content.images(); content.audio(); content.resources(mime_type="application/pdf")
content.artifacts(Summary, category="summary")           # -> tuple[Summary, ...]
content.without_artifacts().without_resources()
content.matching_meta(source="user"); content.split_by_meta(key="section")
```

- `MultimodalContent.of(...)` flattens and normalizes anything `Multimodal`; it raises `TypeError`
  for unsupported values - convert numbers/objects to `str` first.
- Tags: `MultimodalTag.of("text", name="title", meta={...})`, read with `content.tag("title")` /
  `content.tags("title")`, replace with `content.replacing_tag(...)`.
- Passing a pre-built `MultimodalContent` as `input=` is the way to send images/documents.

## Templates

Reusable, storable prompts with `{%placeholder%}` arguments; accepted wherever `instructions` or
`input` is. A bare `{...}` in the body is literal text.

```python
from draive import Template, TemplatesRepository

async with ctx.scope("app", TemplatesRepository.volatile(greet="Hello {%user%}!")):
    prompt = Template.of("greet", arguments={"user": "Ann"})   # immutable handle
    await TextGeneration.generate(instructions=prompt, input="hi")
    await TemplatesRepository.resolve_str(prompt)              # "Hello Ann!"; resolve() keeps multimodal
    await TemplatesRepository.resolve_str(prompt.with_arguments(user="Bob"))
    await TemplatesRepository.resolve_str(Template.of("absent"), default="fallback")
```

- Backends: `TemplatesRepository.volatile(**contents)` (tests), `.file(path)` (JSON), Postgres
  (`integrations.md`), or custom `listing`/`loading`/`defining` callables.
- Errors: `TemplateMissing` (no such template and no `default=`), `TemplateInvalid` (argument
  referenced by the body is not provided, circular template argument, or declared variables do not
  match the body on `define`).
- An argument may itself be a `Template`; nested templates inherit the outer arguments.
- Define: `TemplatesRepository.define(TemplateDeclaration.of(id, variables={...}), content=...)`.

## Tools

A tool is an async function (or async generator) wrapped with `@tool`. The schema is inferred from
the signature; pass tools to a generation call as a list or a `Toolbox`.

```python
from typing import Annotated
from draive import Alias, Description, Toolbox, tool

@tool(description="Return local time", meta={"tags": ("read",)})
async def current_time(location: Annotated[str, Description("City name")]) -> str:
    return f"09:53 in {location}"            # return Multimodal: str, content, resources, artifacts

box = Toolbox.of(current_time, other_tool, suggesting=current_time)   # suggest it on the first turn
await TextGeneration.generate(input="Time in Rome?", tools=box)
```

- `@tool` options: `name=`, `description=`, `parameters=` (explicit JSON schema), `handling=` and
  `meta=`. `handling="response"` (default) returns the result to the model only, `"output"` also
  streams it to the caller's output. `Alias("x")` renames the argument the model sees
  (the key becomes `x`), `Description` documents it.
- A tool can be awaited directly; in the generation loop it runs in the active scope
  (`ctx.state(...)` works).
- Async generator tools yield content parts and `ProcessingEvent`s for streaming progress; events
  are not part of the result the model sees (`GeneratorTool`; `CoroutineTool` for plain async).
- Return values must be multimodal - a `dict`/`int` raises `ToolException`; format it to `str` or
  return an `ArtifactContent`.
- Errors are data for the model, not exceptions for the caller: in the generation loop a failing
  tool becomes a `ModelToolResponse` with `status="error"` and a generic
  `<error>Tool execution failed due to an error</error>` (details only in the logs, never leaked);
  invalid arguments (`ToolArgumentsException`) show the model the argument problem so it can
  retry; an unknown tool name is also an error response. Raise to signal failure, never return
  `"error: ..."` strings.
- `Toolbox.call(name, arguments)` called directly raises `ToolException` (`.tool`, `.meta`);
  `ToolArgumentsException` is its subclass for arguments not matching the parameters.
- Meta: `tool.meta`. Tags drive selection: `box.filtered(tags=("read",))`,
  `box.filtered(tools=("current_time",))`; add with `box.with_tools(t)`; `Toolbox.empty` for none.
- `Toolbox.of(existing_box, ...)` rejects extra tools or `suggesting=`: use `with_tools` /
  `with_suggestion`.

## Skills

File-based instruction packs (agentskills.io layout: `SKILL.md` with frontmatter + resources).

```python
from pathlib import Path
from draive import Agent, ResourceContent, Skill, SkillResource, ctx

async with ctx.scope("skills"):                       # loading reads files through the scope
    skill = await Skill.from_directory(Path("skills/retrieval-assistant"))   # dir name == frontmatter name
agent = Agent.from_skill(skill)                       # instructions + `read_resource(path)` tool
inline = Skill.of(                                    # or build in code
    "demo", description="d", instructions="Do X",
    resources=(SkillResource.of("references/a.md",
               content=ResourceContent.of(b"hi", mime_type="text/markdown")),),
)
skill.has_resource("references/a.md"); skill.resource("references/a.md").content.to_bytes()
tools = [skill.resources_tool()]                      # expose files to an existing toolbox
```

- Frontmatter: `name` (`a-z0-9-`, 1..64, equals directory name) and `description` (1..1024)
  required; `metadata`, `license`, `compatibility`, `allowed-tools` optional; unknown fields are
  ignored. Loading outside a scope raises `ContextMissing`.
- Errors: `SkillLoadingFailed`, `SkillResourceMissing` (both `SkillException`, with `.skill`);
  absolute or `..` resource paths raise `ValueError`.

## Faking the model in tests

Never reach a provider: install a `GenerativeModel` whose `generating` returns an async stream of
chunks. Facades, tools loops and steps run for real on top of it.

```python
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pytest
from draive import (GenerativeModel, ModelContextElement, ModelOutputChunk, ModelOutputStream,
                    ModelTools, ModelToolRequest, TextContent, TextGeneration, ctx, tool)

def fake_model(*turns: Sequence[ModelOutputChunk]) -> GenerativeModel:
    calls: list[Sequence[ModelContextElement]] = []       # inspect what the model was sent
    def generating(*, instructions: str, tools: ModelTools,
                   context: Sequence[ModelContextElement], output: Any = "auto",
                   **extra: Any) -> ModelOutputStream:
        calls.append(tuple(context))
        chunks = turns[min(len(calls) - 1, len(turns) - 1)]    # one entry per loop turn
        async def stream() -> AsyncGenerator[ModelOutputChunk]:
            for chunk in chunks:
                yield chunk
        return stream()
    return GenerativeModel(generating=generating)

@tool
async def weather(city: str) -> str:
    return f"Sunny in {city}"

@pytest.mark.asyncio
async def test_weather() -> None:
    model = fake_model(
        (ModelToolRequest.of("1", tool="weather", arguments={"city": "Oslo"}),),  # turn 1: call tool
        (TextContent.of("sunny"),),                                               # turn 2: answer
    )
    async with ctx.scope("test", model):
        assert await TextGeneration.generate(input="?", tools=[weather]) == "sunny"
```

- `generating` is a plain function returning the stream (not `async def`); `ModelGeneration` wants
  the JSON in a text chunk: `TextContent.of('{"name": "Oslo", "population": 7}')`.
- Skip the model entirely at the facade seam: `ModelGeneration(generating=fn)`,
  `TextGeneration(generating=fn)`, `ImageGeneration(generating=fn)` where `fn` is keyword-only
  `(instructions=..., input=..., toolbox=..., examples=..., **extra)` (`generated` positional first
  for the model one, plus `decoder=`; image/audio take only `instructions`, `input`, `**extra`).
- Fake decisions with `DecisionGeneration(generating=fn)`, where `fn` is
  `async def fn(*, input, questions, **extra)` returning one answer per question in question order,
  with the decided option meta merged like real backends do, e.g.
  `(DecisionChoiceAnswer(identifier="route", choice=billing, meta=billing.meta),)` - the facade
  passes them through unchanged.
- Fake templates with `TemplatesRepository.volatile(...)`.

## Gotchas

- Typed data comes from `ModelGeneration`, not from asking `TextGeneration` for JSON and parsing it.
- A missing `{%x%}` argument fails the call - give defaults in `Template.of(..., arguments=...)`.
- Stopping a raw `GenerativeModel.completion(...)` stream early requires closing it (`ctx.closing`);
  the facades and steps already do.
