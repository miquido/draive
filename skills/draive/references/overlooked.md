# Easily overlooked features - check here before writing it yourself

Draive (through Haiway) ships many small pieces agents tend to reimplement with stdlib or
third-party code. If you are about to write the thing in the left column, use the right one instead.
Everything here is importable from the root `draive` package; verify signatures with `help(...)` or
the docs before use.

## Instead of ... use ...

| About to write | Use |
| --- | --- |
| `os.getenv`, `os.environ[...]`, manual casting | `getenv_str/int/float/bool/base64`, `getenv`, `parse_bool`; `load_env()` reads `.env`; in a `State`, `DefaultEnv("NAME", mapping=int, fallback=...)` or `DefaultEnv("NAME", required=True)` (`fallback` together with `required=True` raises `AssertionError`) |
| `field(default_factory=...)`, `__post_init__` defaults | `Default(value)`, `Default(factory=fn)`, `DefaultFactory(fn, *args)`, `DefaultEnv(...)` |
| hand-written validation in `__init__` | `Annotated[T, Validator(fn)]` (before the built-in validation, coerces raw input) and `Verifier(fn)` (after it, on the typed value); both return the value and raise to reject |
| nested `dataclasses.replace(...)` | `state.updating(...)`; for deep paths `AttributePath`: `User._.profile.name(user)` reads, `User._.profile.name(user, updated="new")` returns an updated copy, `User._.roles[0]` for items |
| JSON/dict (de)serialization, schema | `to_json`, `to_mapping`, `from_json`, `from_json_array`, `from_mapping`, `State.validate(value)`, `State.json_schema()` |
| dropping unknown keys of third party payloads by hand | `from_json(payload, ignore_extra=True)` (also `from_mapping`, `from_json_array`), skipping them at every level |
| renaming/describing fields for APIs and LLM tools | `Alias`, `Description`, `Specification`, `Meta.of(...)` annotations; describe the type itself with `class X(State, description="...")` |
| masking secrets in repr/logs | `Annotated[str, Sensitive()]` |
| `Optional` sentinels, `None` meaning "unset" | `Missing` (`is Missing`), `without_missing(mapping)` to drop unset values |
| `list(x)`, `dict(x)`, `set(x)`, `tuple(x)` of unknown input | `as_tuple`, `as_list`, `as_set`, `as_dict`, `as_map` |
| immutable dict / metadata container | `Map`, `Meta` |
| `asyncio.create_task`, `asyncio.TaskGroup` | `ctx.spawn(...)` (scoped), `ctx.spawn_background(...)` only to outlive the scope |
| `asyncio.gather` | `gather(*coros)` (cancels the rest on failure) |
| semaphore + gather for N at a time | `concurrently(coros, concurrent_tasks=...)`, `execute_concurrently(handler, items, concurrent_tasks=...)`, `process_concurrently(source, handler, ...)` |
| merging async generators | `stream_concurrently(*sources)`, `stream2/3/4_concurrently` (keeps types; by default the merge ends and cancels the rest as soon as any source is exhausted - pass `exhaustive=True` to drain all); close early-left generators with `ctx.closing(gen)` |
| `run_in_executor`, `asyncio.to_thread` | `@asynchronous` (context is propagated; accepts `executor=`) |
| `asyncio.wait_for` | `@timeout(seconds)` |
| retry loops with sleep | `@retry(limit=..., delay=..., catching=...)` - `limit` is the number of retries |
| rate limiting, `asyncio.sleep` spacing | `@throttle(limit=..., period=...)` |
| `functools.lru_cache` on async, manual TTL dicts | `@cache(limit=..., expiration=...)`; shared/remote cache: `cache_externally(make_key=, read=, write=, clear=)` |
| `asyncio.Queue` / async iterator plumbing | `AsyncQueue` (`enqueue`, `finish()` keeps buffered items), `AsyncStream` (`send` waits for the consumer) |
| cursor/offset paging loops | `Pagination.of(limit=..., token=...)` as the argument, `Paginated(items, pagination=...)` as the result |
| string-built URLs | `URITemplate.of("https://{tenant}.x.com/f{/name}{?page}")` - `expand({...})`, `match(uri)`, `matches(uri)` (RFC 6570 subset) |
| `open(...)`, `pathlib` I/O in async code | `Files.access(path, create=...)`, `Files.traverse(path)` (`File`, `Directory`, `FileAccessing`; refuses symlinks, owner-only by default) |
| ad-hoc settings loading | `Configuration` + `ConfigurationRepository` (`.volatile(...)` in tests) |
| raw `httpx`/`aiohttp` client | `HTTPClient` with `draive.httpx.HTTPXClient` |
| raw `asyncpg`, `pika`, ASGI middleware, OpenTelemetry SDK calls | `draive.postgres`, `draive.rabbitmq`, `draive.starlette`/`draive.fastapi`, `draive.opentelemetry` - see `integrations.md` |
| `logging.getLogger`, `print` | `ctx.log_*`, `ctx.record_*`; `setup_logging(...)` once at startup, `JSONLogFormatter`, `LoggerObservability` |
| pub/sub with callbacks or global emitters | `ctx.send(EventState)` / `ctx.subscribe(EventType)` (scope-aware event bus; subscribe before sending) |
| `current_task().cancel()` / manual flags | `ctx.cancel()`, `ctx.check_cancellation()` |
| passing a trace id around | `ctx.trace_id()`, `ctx.trace_context()` |
| module-level singletons / DI container | `ctx.state(Type)`, `ctx.scope(...)`, `ctx.updating(...)`, `ctx.disposables(...)`, `ContextPresets` |
| mock/patch of services in tests | install a fake `State` in `ctx.scope(...)`; explicit state overrides a preset: `ctx.scope(preset, FakeService(...))` |
| `try/finally` around a resource | a disposable (async context manager yielding `State`) |

## Context features people forget

- `ctx.scope(..., kind="server"|"client"|"producer"|"consumer", isolated=True)` - span kind, and
  a task group and event bus of its own.
- `ctx.state(Type, default=...)` and `ctx.contains_state(Type)` for optional dependencies; a
  `State` with no required attribute is built from its defaults when absent.
- `ContextPresets.of(name, *state, disposables=(factories...))` + `with ctx.presets(...)` for
  environments (dev/test/prod) selected by name.
- `ctx.shutdown_background_tasks()` for graceful shutdown and test teardown.
- `ctx.record_*` takes `event=`, or `metric=, value=, kind="counter"|"histogram"|"gauge"`
  (optional `unit=`), each optionally with `attributes=`, or `attributes=` alone. The scope's
  observability level filters what is recorded.
- `@statemethod` gives both `Service.method(...)` (context lookup) and `instance.method(...)`.

## Before adding a dependency to the project

Check this list, `draive.__all__` and the docs first. A new third-party library for retries,
caching, env parsing, async queues, HTTP or Postgres access almost certainly duplicates something
here.
