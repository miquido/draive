# Recipes - common problems and the preferred Draive solution

Each recipe is the approach to reach for first; adapt names to the project. Building blocks
(state, service, resource, wiring) are in `patterns.md`, integration specifics in `integrations.md`.

## Application entrypoint

```python
import asyncio

from draive import LoggerObservability, ctx, load_env, setup_logging


async def main() -> None:
    load_env()                                  # .env into the environment (optional)
    setup_logging("app", debug=False)           # once, at startup; names the loggers to set up
    async with ctx.scope(
        "app",
        AppConfig(),                            # config State (DefaultEnv inside)
        notes_service(),                        # service factories
        disposables=(storage_client(),),        # resources
        observability=LoggerObservability(debug_context=False),   # or omit - logging is the default
    ):
        await run()

asyncio.run(main())
```

Wire everything here, once. Environments (dev/test/prod) are `ContextPresets` chosen by name. For a
web app use `ServerContext` + `application(...)` instead (`integrations.md`).

## Request- or job-scoped data

Open a nested scope per unit of work and add what is specific to it; it inherits the rest.

```python
async def handle(job: Job) -> None:
    async with ctx.scope("job", JobContext(job_id=job.id)):
        await process(job)
```

Use `with ctx.updating(SomeState(...)):` to override state for a block without a new scope.

## Optional dependency

```python
metrics = ctx.state(Metrics, default=Metrics.noop())   # used when Metrics is not in the scope
if ctx.contains_state(Cache): ...                       # or check explicitly
```

Without a default, an absent `State` is built from its own defaults when it has no required
attribute; otherwise `ctx.state(...)` raises `ContextStateMissing` - a wiring error, not something
to catch.

## Per-call override of configuration

Declare `**extra: Any` in the Protocol and the function, resolve config, override for this call:

```python
async def store(content: str, **extra: Any) -> None:
    path = ctx.state(NotesDirectory).updating(**extra).path
```

## Many things at once

| Need | Use |
| --- | --- |
| a few independent calls, all must succeed | `a, b = await gather(fa(), fb())` |
| same handler over many items, N at a time | `await execute_concurrently(handler, items, concurrent_tasks=8)` |
| many coroutines, bounded, keep failures | `await concurrently(coros, concurrent_tasks=8, return_exceptions=True)` |
| consume a source, no results | `await process_concurrently(source, handler, concurrent_tasks=8)` |
| work alongside the current task | `task = ctx.spawn(fn, arg)`; `await task` or let the scope join it |
| work that must survive the scope | `ctx.spawn_background(fn, arg)` - no scope state, pass what it needs |

`concurrent_tasks` defaults to 2. Never leave tasks unreferenced or reach for `asyncio.create_task`.

## Resilient adapter around an unreliable call

Decorate the adapter function, not the callers. The outer decorator wraps the inner ones.

```python
@cache(limit=256, expiration=60)      # skip repeats - keyed on arguments only, never on `ctx`
@timeout(5)                           # bounds the whole call, retries included
@retry(limit=3, delay=0.2, catching=TransientError)   # up to 3 retries: 4 attempts
@throttle(limit=10, period=1)         # stay within provider rate limits
async def fetch_rate(currency: str) -> Decimal: ...
```

`limit` counts retries after the first attempt; `delay` may be a function of `(attempt, exception)`;
`catching` a type or a predicate. Retry only errors that are safe to repeat; never retry a consumed
HTTP body. Put `@timeout` under `@retry` to bound each attempt instead.

## Parsing external input

```python
try:
    command = Command.from_json(payload)
except ValueError as exc:                          # malformed JSON and invalid content alike
    raise BadRequest(...) from exc
```

`from_json` raises `ValueError`; `from_mapping`, `State.validate(value)` and the constructor raise
`ValidationError` (not a `ValueError`) naming the failing path. Put format rules into the type
(`Validator` to coerce before validation, `Verifier` to enforce after, `Literal`, unions) so invalid
data cannot get past the boundary. Return data with `to_json()`/`to_mapping()`.

## Modelling outcomes and variants

Use a union and `match`; do not return `(ok, value, error)` tuples or flags.

```python
type Result = Success | Failure

match result:
    case Success(value=value): ...
    case Failure(reason=reason): ...
```

Add a variant and the type checker points at every `match` to update.

## Evolving a State

Add fields with defaults so existing construction and stored JSON keep working; rename on the wire
with `Alias`; use `Missing` for "not provided" distinct from `None`; derive instead of adding a
second field that duplicates another.

## Streaming results

```python
async def rows() -> AsyncGenerator[Row]:
    async for message in source():
        yield transform(message)

async with ctx.closing(rows()) as stream:       # always close early-left generators
    async for row in stream:
        ...
```

A generator that needs a scope keeps it inside its body. Merge sources with `stream_concurrently`.
For HTTP responses use `StreamResponse` so the request scope stays open until the last chunk.
Bridge a callback-style producer with `AsyncQueue` (`enqueue(...)`, `finish()` ends it, buffered
items are still delivered).

## Notifying other parts of the app

```python
class OrderPlaced(State):
    order_id: UUID

async def audit(events: AsyncIterator[OrderPlaced]) -> None:
    async for event in events:
        ...

ctx.spawn(audit, ctx.subscribe(OrderPlaced))        # subscribed here, before anything is sent
...
ctx.send(OrderPlaced(order_id=...))                 # to subscribers in this scope and enclosing ones
```

A subscription exists from the `ctx.subscribe(...)` call and sees only events sent afterwards, so
create it before spawning the consumer - subscribing inside the spawned task loses events sent
before the task first runs. Dispatch is by exact type; events nobody subscribed to are dropped.
`broadcast=True` reaches every subscriber of the bus regardless of the scope tree; a scope opened
with `isolated=True` has a bus of its own. The subscription ends when the scope that created it
exits, so consume it in a task of that scope.

## Calling sync or blocking code

```python
@asynchronous
def hash_password(raw: str) -> str: ...

digest = await hash_password(raw)          # runs in an executor, context preserved
```

Never call blocking code directly in a coroutine, and do not start threads yourself.

## Testing

| Situation | Approach |
| --- | --- |
| a service's collaborators | `ctx.scope("test", Service(fn=fake), Dependency(fn=fake))` |
| code using `HTTPClient` | `HTTPClient(requesting=fake_request)` returning `HTTPResponse(...)` (`integrations.md`) |
| `Configuration` | `ConfigurationRepository.volatile(MyConfig(...))` |
| replacing one piece of a preset | `ctx.scope(preset, Fake(...))` - explicit state wins |
| per-test isolation | everything inside `ctx.scope(...)`; autouse fixture calling `ctx.shutdown_background_tasks()` after each test |
| several inputs | `pytest.mark.parametrize`, not loops |
| async code | `@pytest.mark.asyncio`, no `sleep` for synchronisation - await the task or event |

Assert on returned states and recorded calls; do not patch modules or globals.

## Migrating existing code

1. Turn data classes and dicts into `State` (annotate everything, collections abstract).
2. Replace globals/singletons and constructor injection with `ctx.state(...)` lookups.
3. Extract each external call behind a `Protocol` held in a service `State`.
4. Replace `os.getenv`, `asyncio.create_task`/`gather`, retry/caching loops, `print`/`logging` with
   the helpers in `overlooked.md`.
5. Move resource opening into disposables and wire everything in one entrypoint scope.
6. Add fakes and tests for each service before refactoring further.
