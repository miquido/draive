# Workflows - steps, agents, conversation, resources, guardrails

All names import from `draive` unless noted. Every provider-facing call needs a model in scope
(a provider's config state + its `disposables`, see `generation.md`); tests inject
`GenerativeModel(generating=fake)` with `fake(*, instructions, tools, context, output, **extra)`
returning an async iterable of chunks.

## Steps - `Step`, `StepState`

`StepState` is immutable: `context` (`ModelContext`) + `artifacts` (typed `State` values stored by
class name, or by keyword key). `Step` transforms it and streams output chunks. Composition returns
a new `Step`; nothing runs until you execute it inside `ctx.scope`.

```python
class Counter(State):
    n: int = 0

@step                                              # plain StepState -> StepState coroutine
async def bump(state: StepState) -> StepState:
    n = state.get(Counter, default=Counter()).n
    return state.updating_artifacts(Counter(n=n + 1))

async def below_three(state: StepState, iteration: int) -> bool:
    return state.get(Counter, default=Counter()).n < 3

pipeline = Step.emitting("hi") >> Step.loop(bump, condition=below_three)   # >> == .then == sequence

content = await pipeline.run()              # MultimodalContent of all emitted parts
state = await pipeline.process()            # final StepState
state.get(Counter, required=True)           # also key="name"; default=...
async for chunk in pipeline.stream(): ...   # output chunks only (no state)
```

Execution args: `run/process/stream(context=(), /, *artifacts, **keyed_artifacts)` or a ready
`StepState`. Artifacts given in one call must have unique keys (`ValueError`).

Builders (classmethods): `Step.sequence(*steps)`, `Step.loop(*steps, condition=, max_iterations=32)`
(condition is `async (state, iteration) -> bool`), `Step.concurrent(*steps, merge=)` (`merge` is
`async (branches: Iterable[StepState]) -> StepState`), `Step.selection(async (state) -> Step)`,
`Step.route(criteria, *steps, input=)` (the scope `DecisionModel` picks one step, offered by the
meta `name`/`description` set with `step.with_meta(name=, description=)`; no implicit fallback to
completions - bind a backend; a missing/repeated name raises `ValueError` when built, an unknown
answer raises `StepException`; wrapping `with_*` modifiers keep meta, `sequence`/`then` do not),
`Step.emitting(...)`, `Step.updating_artifacts/removing_artifacts`, `Step.appending_input/output`,
`Step.appending_context/replacing_context/mutating_context`,
`Step.preserving_state/restoring_state`, `Step.generating_completion(instructions=, tools=, input=,
output=)` (one model call), `Step.looping_completion(instructions=, tools=, input=, output=,
max_iterations=32)` (model + tool loop), `Step.handling_tools(tools)`,
`Step.decoding_artifact(Report)` (decode structured output into an artifact; request it with
`output=Report` on the completion).

Modifiers (instance methods, all return a `Step`; all but `with_meta` keep the step meta):
`with_meta(name=, description=)` (describes the step, `Step.meta`), `with_ctx(*state, disposables=)`,
`with_retry(limit=1, delay=, catching=)`, `with_fallback(step, catching=)`, `with_timeout(seconds)`,
`with_condition(bool | async pred, alternative=step)`, `with_isolated_context`,
`with_volatile_context()`, `with_volatile_tools()`, `with_suppressed_output()`,
`with_context_evaluation(...)`, `with_output_evaluation(...)`.

Failures carry the state they happened with (`exc.state`):

| Exception | When |
| --- | --- |
| `StepException` | loop / `looping_completion` exceeds `max_iterations`, decoding fails, evaluation fails with `raise_on_failure` |
| `StepTimeout` | `with_timeout` exceeded (`exc.timeout`); time the consumer spends between chunks is not counted |
| `StepRetriesExhausted` | retried step still failing (`exc.attempts`; last error is `__cause__`) |

Gotchas:
- `max_iterations=None` disables the loop limit; `< 1` raises `ValueError`.
- A retried step restarts from the *original* input state (partial state is discarded) but chunks
  already emitted cannot be retracted. Errors not matched by `catching` are raised unchanged.
- Close a stream you stop consuming early (`ctx.closing`); leaving it to the GC finalizes it in
  the wrong context.
- Type custom callbacks with `StepExecuting`, `StepMerging`, `StepSelecting`, `StepProcessing`,
  `StepDecisionInputPreparing` (exported from `draive`/`draive.steps`).

## Agents - `Agent`, `AgentsGroup`, `AgentMemory`

`Agent` is an immutable wrapper: identity (`AgentIdentity`: `uri`, `name`, `description`, `meta`)
+ an executor `AgentMessage -> AsyncGenerator[MultimodalContentPart | ProcessingEvent]`. It holds
no conversation; state lives in `AgentMemory` and the scoped `AgentThread`.

```python
worker = Agent.steps(                                # deterministic pipeline
    memory.recall_step(), reply_step, memory.remember_step(),
    agent="worker", description="Does X",            # str name or AgentIdentity
)
assistant = Agent.generative(                        # model + tools loop
    agent="support", description="...", instructions="...",   # str | Template
    tools=[some_tool], memory=AgentMemory.volatile(threads_limit=32),
    output="auto", max_iterations=32,
)
skilled = Agent.from_skill(skill, tools=..., memory=...)      # a loaded Skill (`generation.md`)

async with ctx.scope("app", *model_state):
    async for chunk in assistant.call(input="Hi", thread=thread_uuid):   # thread/meta optional
        ...
    # or with a prepared message: assistant.respond(AgentMessage.of("Hi", thread=uuid))
```

- `Agent.steps` seeds the context with the message as one `ModelInput`, runs the steps as a
  sequence and drops reasoning and tool request/response chunks - callers only see content and
  `ProcessingEvent`s. It applies **no** memory implicitly.
- `Agent.generative` / `from_skill` call `memory.prepare`, `recall`, `remember` themselves; a
  tool-requesting model past `max_iterations` raises `AgentException`.
- `call` reuses the active `AgentThread` (nested calls share the thread, `meta` is merged);
  `AgentThread.of(identifier, *, agent_uri=, meta=)` - both identifier and `agent_uri` are
  required. `Agent.noop(identity)` is a placeholder implementation.

### AgentMemory

Context is stored as an immutable full snapshot per `(agent_uri, thread)`; latest wins, and
whatever is remembered becomes the next recalled context verbatim, so steps may compact or replace
context freely. Concurrent remembers on one thread are last-writer-wins.

Contract (all receive the `AgentThread`, plus `**extra`):
- `prepare(thread, instructions) -> Sequence[Tool] | None` - runs before each recall, must be
  idempotent per thread; returned tools live for that turn and replace same-named agent tools.
- `recall(thread, context) -> ModelContext` - gets the full turn context so far, returns stored
  history + it.
- `remember(thread, context) -> None` - persists the context produced by the turn.

```python
AgentMemory.volatile(initial=(), threads_limit=None)   # in-process; LRU refreshed by prepare/remember, not recall
AgentMemory(recalling=..., remembering=..., preparing=None)   # custom backend
memory.with_ctx(SomeState(), disposables=(client(),))         # bind state around memory calls
AgentMemory.disabled                                          # default: context lives for one turn
```
Persistent memory is keyed by `agent_uri`: a string `agent="name"` gets a random
`agent://<uuid>` per process, so pass a stable `AgentIdentity.of(uri="agent://support", name=...)`
or history is not found after a restart.
Protocols `AgentMemoryPreparing/Recalling/Remembering` import from `draive.agents`. Postgres:
`PostgresAgentMemory.instance()` from `draive.postgres`.

Memory steps (`recall_step(**extra)`, `remember_step(**extra)`) need an `AgentThread` in scope
(agents bind one; otherwise `AgentException`). Gotchas:
- Recalling twice in one pipeline duplicates history. There is no `prepare_step` - preparation (and
  memory-contributed tools) only works in `generative`/`from_skill`.
- Remember happens only after the output stream is fully consumed; abandoned or failed turns are
  not persisted.
- Requests/responses of memory-provided tools are stored in the context - keep tool names stable.

### Delegation

```python
group = AgentsGroup.of(researcher, reviewer)          # *Agent | AgentIdentity; unique names and URIs (ValueError)
coordinator = Agent.generative(
    agent="coordinator", instructions="Delegate research, then answer.",
    tools=[group.as_tool()],                          # tool `agent_request(agent: name, task: str)`
)
researcher.as_tool(handling="output")                 # single agent as tool; name=/description=/meta=
async for chunk in group.call("researcher", input="..."): ...   # name or URI
```

`handling="response"` (default) feeds the result back to the caller's loop; `"output"` streams the
delegate's output straight to the user. An `AgentIdentity` in the group is a placeholder: calling it
raises `AgentUnavailable` until `group.bind(agent)` (same URI; unknown URI -> `AgentException`).
Recreate generated tools after renaming agents.

## Conversation

```python
async with ctx.scope("chat", *model_state):
    stream = Conversation.completion(          # classmethod; or Conversation().completion(...)
        instructions="...",                    # str | Template
        message="Hi",                          # Multimodal | Template
        tools=[a_tool],                        # Toolbox | Iterable[Tool]
        memory=ConversationMemory.volatile(),  # or Sequence[ConversationTurn] (read-only history)
    )
    async for chunk in stream: ...             # content parts, reasoning chunks, ConversationEvent
```

- Output chunk is `MultimodalContentPart | ModelReasoningChunk | ConversationEvent` (tool
  requests/responses/events arrive as `ConversationEvent`). The tool loop is capped at 32 rounds
  (not configurable) -> `ConversationException`.
- `ConversationMemory(fetching=, recalling=, remembering=)`: `fetch(pagination)` pages of
  `ConversationUserTurn`/`ConversationAssistantTurn`, `recall(pagination=None) -> ModelContext`,
  `remember(*turns)` (a user + assistant turn is stored per completion). `.disabled` is the
  default, `.constant(turns)` replays fixed history and ignores writes, `.volatile()` accumulates
  in-process and supports no pagination tokens.
- Realtime: `RealtimeConversation.prepare(instructions=, tools=, memory=, output="auto")` returns
  an async context manager yielding a `RealtimeConversationSession` (needs a realtime model):

```python
async with RealtimeConversation.prepare(instructions="...") as session:
    await session.write(TextContent.of("Hello"))   # ConversationInputChunk
    async for chunk in session.reader(): ...       # loops until the session errors/closes
    # also session.read(), session.writer(input_stream)
```

## Resources

Addressed by URI. `ResourceReference.of(uri, name=, description=, mime_type=)` points at one;
`ResourceContent.of(bytes | base64 str, mime_type=)` holds data (`to_bytes()`, `to_data_uri()`;
`to_str()` is a markdown placeholder, not the decoded text); `Resource.of(content, uri=,
mime_type=)` binds uri + content (or a collection of references for a directory), read back as
`resource.resource`; `resource.reference` derives the reference.

`ResourcesRepository` (State, wire it in scope) delegates to four replaceable functions:

```python
ResourcesRepository(
    list_fetching=..., fetching=..., uploading=..., deleting=...,   # ResourceListFetching etc.
)
res = await ResourcesRepository.fetch("https://x/y", required=False)   # Resource | None
await ResourcesRepository.fetch(ref, default=fallback)                  # or required=True (raises ResourceMissing)
await ResourcesRepository.upload(Resource.of(b"..", uri=..., mime_type="text/plain"))   # -> Meta
await ResourcesRepository.fetch_list(pagination=None)                   # Paginated[ResourceReference]
await ResourcesRepository.delete(res)
```

The default repository speaks HTTP(S) only (`ResourceUnresolveable` for other schemes) and has no
listing (`fetch_list` raises `NotImplementedError`). Uploading a `Resource` that holds references
raises `ResourceCorrupted`. Errors: `ResourceException` > `ResourceMissing`, `ResourceCorrupted`,
`ResourceInaccessible`, `ResourceUnresolveable`.

Resource templates declare an RFC 6570 URI and resolve through an async function:

```python
@resource(uri_template="https://api.example.com/{org}/issues/{issue_id}{?state}")
async def issue(org: str, issue_id: int, state: str = "open") -> ResourceContent:
    return ResourceContent.of(f"{org}#{issue_id}".encode(), mime_type="text/plain")

res = await issue.resolve(org="acme", issue_id=1)                       # -> Resource
res = await issue.resolve_from_uri("https://api.example.com/acme/issues/1?state=closed")
issue.matches_uri(uri)   # bool, no resolving; issue.available (availability_check=)
```

Every template variable must match an argument by name or `Alias`, else `ValueError` at declaration;
values from a URI are coerced to the annotated types. `@resource` also takes `mime_type=`, `name=`,
`description=`, `availability_check=`, `meta=`. Matching is strict: variables span one non-empty
segment, traversal (`..`), control chars and (for templates without `{?...}`) undeclared query
parameters are refused - use it as the guard before fetching. A failing resolver surfaces as
`ResourceCorrupted`. `URITemplate.of(...)` (`expand`, `match`, `variables`) works standalone.

## Guardrails

Four State-backed checks, each a `statemethod` (call on the class with the instance in scope, or
on an instance), all accept `Multimodal` and `**extra`, and report a verdict metric:

| State | Call | Raises |
| --- | --- | --- |
| `GuardrailsModeration(input_checking=, output_checking=)` | `check_input(c)`, `check_output(c)` | `GuardrailsInputModerationException` / `GuardrailsOutputModerationException` (`.violations: Mapping[str, float]`, `.content`, `.replacement`) |
| `GuardrailsSafety(sanitization=)` | `sanitize(c) -> MultimodalContent` | `GuardrailsSafetyException` (`.reason`, `.content`) |
| `GuardrailsAnonymization(anonymizing)` | `anonymize(c) -> GuardrailsAnonymizedContent` (`.content`, optional `await .deanonymize(c)`) | - |
| `GuardrailsQualityVerification.of(evaluator)` | `verify(c)` | `GuardrailsQualityException` (`.reason`, `.content`) |

```python
async def no_bad_words(content: MultimodalContent, /, **extra: Any) -> None:
    if "bad" in content.to_str():
        raise GuardrailsModerationException(
            "blocked", violations={"harm": 0.9}, content=content,
            replacement=MultimodalContent.of("[removed]"),
        )

with ctx.updating(GuardrailsModeration(input_checking=no_bad_words)):
    await GuardrailsModeration.check_input(user_text)     # catches -> handle / use .replacement
```

- A checker signals a violation by raising the base `GuardrailsModerationException` (it is re-raised
  as the input/output subtype); any other exception becomes `GuardrailsFailure` (`.cause`) - the
  check itself failed, distinct from "content rejected". Both derive `GuardrailsException`
  (`.meta`). An unconfigured `GuardrailsModeration` allows everything.
- `GuardrailsSafety()` defaults to `guardrails_regex_sanitizer()` whose rules target prompt
  injection (ignore-instructions, system-prompt exfiltration, role override, encoded instructions),
  not generic PII/secrets. Custom rules:
  `guardrails_regex_sanitizer(rules=(RegexSanitizationRule(identifier=, pattern=re.compile(..),
  reason=, mask="***" | None, validator=None),))` - `mask=None` blocks (raises) instead of masking;
  identifiers must be unique.
- `GuardrailsQualityVerification.of(...)` takes a prepared evaluator or evaluator scenario
  (`providers-and-data.md`); an errored evaluation is a `GuardrailsFailure`, not a quality verdict.
- Guardrails are explicit calls - nothing in steps/agents/conversation invokes them. Call
  `check_input` before and `check_output` after the model. Provider-backed moderation is the
  `GuardrailsModeration` installed by the `OpenAI()`, `Mistral()` and `Bedrock()` adapters.
