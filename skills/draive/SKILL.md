---
name: draive
description: Write, review and refactor Python code that uses the Draive framework (`from draive import State, ctx, ...`) for Gen-AI applications - immutable State types, Protocol-based services, contextual dependencies through ctx.scope, structured concurrency, observability, plus the Draive layer - model providers (OpenAI, Anthropic, Gemini, Mistral, Ollama, Bedrock...), text/structured/image/audio generation, tools, skills, steps, agents and memory, conversations, resources and templates, MCP, embeddings and vector stores, evaluation, guardrails. Use whenever a project imports draive, or when asked to add a model call, tool, agent, pipeline, evaluator or integration to such a project, or to design an application "the Draive way".
---

# Writing code with Draive

Draive (Python 3.15+) is a framework for high-quality Gen-AI applications. It is built on Haiway and
re-exports it: `from draive import State, ctx, statemethod, ...` is the same API as `haiway`, so
everything below about state, scopes and concurrency applies unchanged. Draive adds the Gen-AI
layer (models, generation, tools, steps, agents, evaluation...) on the same three primitives:
immutable `State`, the scoped context `ctx`, and structured concurrency. There is no DI container
and no mutable service object: behavior is `Protocol`-typed functions held inside a `State` and
resolved from the current scope.

Follow the existing code of the project first; use this skill for what it does not already settle.
When unsure about an API, read the installed package (`draive.__all__`, docstrings, `help(...)`) or
the docs at https://miquido.github.io/draive/ (and https://miquido.github.io/haiway/ for the
underlying primitives) rather than guessing - names change between versions.

## Principles to design by

1. **Make illegal states unrepresentable.** Prefer unions, closed `Literal`s and small value types
   over optional fields valid only together, or flags meaningful in some states only.
2. **One source of truth.** Never store what can be computed; never repeat a name, value or list
   that already has a home.
3. **Small, composable interfaces.** One narrow responsibility per `Protocol`, type or function;
   combine by composition, not by growing one interface.

## Rules that are easy to get wrong

**Types and data**

- Subclass `State` for every domain, config and service object. It is immutable: change through
  `.updating(...)`, never by assignment.
- Annotate every attribute with abstract collections - `Sequence`/`Mapping`/`Set` - never
  `list`/`dict`/`set`. Whatever is written, values are stored as `tuple`/`Map`/`frozenset`.
- A `State` declaring attributes is final and has a single base: it cannot be subclassed, mixed in
  or combined. Reuse by holding an instance as an attribute, by parametrizing a generic
  (`class Box[T](State)`), or by inheriting from an attribute-less `State` that only adds behavior.
- Newest typing syntax: `type` aliases, PEP 695 generics, `X | None`. No loose `Any` except at
  third-party boundaries; no `cast` unless provably safe and local. Narrow unions with
  `match`/`isinstance`.
- `Missing` is the "not provided" sentinel, distinct from `None`: test with `is Missing`, never by
  truthiness (it is truthy); `case Missing:` captures instead of comparing.
- Mark secrets `Annotated[str, Sensitive()]` (redacted in `repr`/`str`, still present in
  `to_json`/`to_mapping`); read them from the environment with `DefaultEnv("NAME", required=True)`
  or `getenv_str`, never `os.getenv`.

**Services and dependencies**

- A contract is a `Protocol` named as a gerund (`UserFetching`), fully typed, `async` for I/O.
- An implementation is a plain function; a service is a `State` holding those functions and
  exposing `@statemethod` entry points. Dependencies are resolved with `ctx.state(Type)` inside the
  implementation - not passed through constructors, not stored in globals.
- `ctx.state(Type)` looks up the exact type. When it is absent, a `State` constructible without
  arguments is built from its defaults; otherwise `ContextStateMissing` is raised.
- Install state with `ctx.scope(...)`; external resources (clients, pools, files) are
  **disposables** so cleanup always runs. Tests install fakes the same way.
- Layout per functionality: `types.py` (Protocols + payload State), `state.py` (service State +
  statemethods), `implementation.py` (functions + factory). See `references/patterns.md`.

**Concurrency**

- All I/O is async. Never block the event loop; bridge unavoidable sync calls with
  `@asynchronous`, never with custom threads.
- Start tasks with `ctx.spawn(...)` (scoped, joined by the scope on exit). Use
  `ctx.spawn_background(...)` only when work must outlive the scope - it has no scope state, so
  `ctx.state(...)` raises `ContextMissing` there: pass what it needs or open a scope inside.
- Use `gather(...)`, `concurrently(...)`, `execute_concurrently(...)` instead of raw
  `asyncio.gather`/`create_task`. Wrap async generators left early in `ctx.closing(...)`.

**Errors**

- Translate provider/SDK exceptions into typed ones at the boundary; keep context (provider,
  operation, identifiers), redact sensitive payloads. Never raise bare `Exception`.
- Misuse of an API is a programming error and is checked by assertions; expected runtime failures
  are typed exceptions.

**Observability**

- Never `print` or use `logging` directly. Use `ctx.log_debug/info/warning/error` and
  `ctx.record_debug/info/warning/error`. A record is an `event=`, or a metric
  (`metric=, value=, kind=`, optional `unit=`), each optionally with `attributes=`, or
  `attributes=` alone - never an event and a metric in one call. There is no `ctx.record(...)`.
- Never log secrets or full request bodies.

## The Draive layer

Gen-AI work follows the same rules: models, tools, memory and stores are `State`s installed in a
scope, called through their `@statemethod`s, replaceable by fakes in tests.

- Pick the facade by the job: text/structured/image/audio generation, a `Step` pipeline, an `Agent`,
  a conversation, an evaluator. Do not call a provider SDK directly - go through the provider
  adapter (`draive.openai`, `draive.anthropic`, ...) installed as a disposable of the scope.
- Structured output is a `State` type (its schema is generated from the type); give it field
  `Description`s rather than prose in the prompt.
- Tools are typed functions (`@tool`) collected in a `Toolbox`. A tool failure reaches the model as
  an error result: invalid arguments show the model their message so it can correct the call, any
  other exception only as a generic "Tool execution failed" - details stay in the logs, so never
  rely on an exception message to steer the model.
- Provider extras are optional: `uv add "draive[openai]"`; importing an adapter without its extra
  fails at import time.
- Unit tests never reach a provider: install a fake model/`State` in `ctx.scope`.
- Evaluation lives in `draive.evaluation` / `draive.evaluators` and helpers such as
  `VolatileVectorIndex` in `draive.helpers` - not in the root package.

Read the matching reference (listed at the end) before writing code in an area.

## Don't reinvent - check before writing

Draive (through Haiway) already ships helpers agents tend to skip: `@retry`, `@timeout`,
`@throttle`, `@cache`, `@asynchronous`, `gather`/`concurrently`/`stream_concurrently`, `AsyncQueue`,
`getenv_*`/`DefaultEnv`, `Validator`/`Verifier`, `AttributePath`, `Files`, `Configuration`,
`URITemplate`, `Paginated`, the event bus, and the extras (`httpx`, `postgres`, `rabbitmq`,
`starlette`, `fastapi`, `opentelemetry`).
Scan `references/overlooked.md` before writing retry loops, env parsing, caching, queues, URL
building, file access or a client wrapper, and before adding a third-party dependency.

## Workflow

1. Look at how the project already splits functionalities and names things; match it. Put new code
   in the right layer: entrypoints -> features -> solutions -> integrations -> commons, where a
   layer depends only on those after it (features never on integrations directly).
2. Model the data as `State` first, then the `Protocol`s, then implementations, then wiring.
3. Wire dependencies at the edges (application entry, request handler, test) with `ctx.scope`.
4. Write tests with `pytest.mark.asyncio`; isolate state with `ctx.scope(...)`, install fakes as
   state, no network.
5. Run the project's formatter, linter, type checker and tests (ruff, bandit, pyright strict,
   pytest - see `references/project.md`) and fix findings instead of silencing them; do not loosen
   types or add ignores.
6. Public symbols get NumPy-style docstrings (Parameters/Returns/Raises); private helpers none.

## Reviewing Draive code - checklist

- Any `list`/`dict`/`set` annotation, mutable shared container, `print`, `logging`, `os.getenv`,
  `asyncio.gather` or `create_task`, blocking call in `async def`, global singleton or hand-passed
  dependency?
- Service logic reaching for a client directly instead of a `Protocol` held in `State`?
- Invalid combinations of fields that a union would forbid? Values stored that could be derived?
- Resources opened without a disposable? Background task relying on `ctx.state(...)`?
- An event subscription created after the events it should see may already be sent?
- Secrets without `Sensitive()`, or logged?
- Exceptions from a provider leaking through untranslated?
- A hand-written helper that `references/overlooked.md` already lists?
- A provider SDK called directly, a model faked by patching instead of installed state, a prompt
  asking for JSON where a structured `State` type would do?

## References

Read the one that matches the task; each is self-contained.

- `references/generation.md` - provider wiring, text/structured/image/audio generation, multimodal
  content, templates, tools and skills, faking a model in tests.
- `references/workflows.md` - steps, agents and memory, conversations, resources, guardrails.
- `references/providers-and-data.md` - provider extras and gotchas, MCP, embeddings, vector stores,
  datastores, AWS (S3, SQS, CloudWatch), evaluation, web and telemetry wiring through `draive.*`.
- `references/patterns.md` - the building blocks with code: modelling `State`, configuration,
  services, disposable resources, features, wiring with scopes and presets, tasks, errors, tests.
  Read it when adding or changing a state, service, resource or feature.
- `references/recipes.md` - preferred solutions for recurring problems: entrypoint, scoped data,
  optional dependencies, concurrency choices, resilient adapters, parsing input, streaming, events,
  sync bridging, testing situations, migrating existing code.
- `references/integrations.md` - HTTP, Postgres, RabbitMQ, Starlette/FastAPI, OpenTelemetry,
  configuration and files: wiring, usage and the gotchas of each. Read it before touching one.
- `references/overlooked.md` - "instead of X use Y" table of helpers that are easy to skip. Read it
  before writing any utility.
- `references/style.md` - immutability, typing, functional style, errors, logging and documentation
  conventions in full. Read it before a larger piece of work or a review.
- `references/project.md` - package layout and dependency direction, pyproject, ruff/pyright/pytest
  setup, make targets. Read it when creating a project or a new package.
