# Integrations

Install the extra and import from the subpackage: `draive[httpx]` -> `draive.httpx`, and likewise
`postgres`, `rabbitmq`, `starlette`, `fastapi`, `opentelemetry` (re-exports of the Haiway
integrations). Provider, MCP and vector-store integrations are in `providers-and-data.md`. Each one
installs a client as a disposable and exposes a `State` with `@statemethod` calls. The contracts
(`HTTPClient`, `MQQueue`, `MQMessage`, `Files`, `Configuration`) live in the root package: use those
in your own signatures so code stays implementation-agnostic, and keep integrations behind a
solution-level service so features never see them.

Prefer these over raw `httpx`, `asyncpg`, `pika`, middleware of your own or direct OpenTelemetry
calls - they already integrate scopes, tracing, error types and cleanup. For exact signatures read
the Haiway docs page of the integration (https://miquido.github.io/haiway/) or the installed
package.

## HTTP - `draive.httpx`

```python
from draive import HTTPClient, ctx
from draive.httpx import HTTPXClient

async with ctx.scope("app", disposables=(HTTPXClient(base_url="https://api.example.com"),)):
    response = await HTTPClient.get(url="/users", query={"page": 1})
    if response.status_code != 200:              # 4xx/5xx are returned, not raised
        raise UsersUnavailable(status=response.status_code)
    users = json.loads(await response.body())
```

- `get`/`post`/`put`/`request(method, ...)`; all arguments but the method are keyword-only.
- Raised: `HTTPTimeoutError`, `HTTPConnectionError`, `HTTPBodyConsumedError`, all `HTTPClientError`.
- The body is read up front unless `stream=True`; a streamed body (`response.stream_body()`) is read
  once and inside the scope that issued the request.
- `trace_propagation=True` per request, only for services you own.
- Fake for tests: a function with the `HTTPRequesting` signature held in `HTTPClient`:

```python
async def fake_request(
    method: str, /, *, url: str, query: HTTPQueryParams | None = None,
    headers: HTTPHeaders | None = None, body: HTTPBody | None = None,
    timeout: float | None = None, follow_redirects: bool | None = None, stream: bool = False,
) -> HTTPResponse:
    return HTTPResponse(status_code=200, headers={}, body=b"{}")

async with ctx.scope("test", HTTPClient(requesting=fake_request)):
    ...
```

## Postgres - `draive.postgres`

```python
from draive.postgres import Postgres, PostgresConnection, PostgresConnectionPool

async with ctx.scope("app", disposables=(PostgresConnectionPool(),)):   # POSTGRES_* variables
    row = await Postgres.fetch_one("SELECT email FROM users WHERE id = $1", user_id)
    email = None if row is None else row.get_str("email", required=True)

    async with ctx.disposables(Postgres.acquire_connection()):      # pin one connection
        async with PostgresConnection.transaction():                # rolls back on exception
            await PostgresConnection.execute("DELETE FROM jobs WHERE finished")
            await PostgresConnection.execute("INSERT INTO audit(action) VALUES($1)", "cleanup")
```

- `fetch`, `fetch_one`, `execute` on `Postgres` (a pooled connection per call) and on
  `PostgresConnection` (the pinned one). `PostgresRow` has typed getters: `get_str`, `get_int`,
  `get_uuid`, `get_datetime`, ... with `default=`/`required=`.
- A nested `transaction()` is a savepoint. `acquire_connection()` cannot be used where a
  connection is already pinned.
- Nothing is retried by default. `retries=N` repeats one statement on a serialization failure or
  deadlock - valid only when the statement is its own transaction. To retry a transaction, loop
  around the whole `transaction()` block.
- The pool reads `POSTGRES_HOST/PORT/DATABASE/USER/PASSWORD/SSLMODE/CONNECTIONS/...`.
  `POSTGRES_CONNECTIONS` defaults to **1**: raise it, or a pinned connection blocks every other
  query. `PostgresConnectionPool.of(dsn)` ignores those variables.
- `execute` with arguments runs one statement; without arguments it may run a script.
- Errors are `PostgresException` (`PostgresErrorCode`). `Postgres.execute_migrations(...)` runs
  migrations; `PostgresConfigurationRepository.prepare(...)` backs `Configuration` with a table.

## RabbitMQ - `draive.rabbitmq`

```python
from draive.rabbitmq import RabbitMQ, RabbitMQClient

async with ctx.scope("worker", disposables=(RabbitMQClient(),)):
    async with await RabbitMQ.queue("orders", content_encoder=encode, content_decoder=decode) as queue:
        await queue.publish(order)
        async with await queue.consume() as messages:      # leaving the block cancels the consumer
            async for message in messages:
                if message.meta.get("attempt", 1) > 3:
                    await message.reject(requeue=False)    # stop a poison message
                    continue
                async with message as content:             # ack on success, reject on exception
                    await handle(content)
```

- Both `RabbitMQ.queue(...)` and `queue.consume()` are awaited **and** entered: `async with await`.
- A rejected message is requeued by default, so a handler that always fails loops - guard with
  `attempt` as above (declare the queue `x-queue-type: quorum` for a real count).
- Handlers must be idempotent: consumers are re-established after a connection loss and in-flight
  messages are delivered again.
- `RabbitMQ.declare_queue/delete_queue/purge_queue` manage queues; publisher confirms and trace
  propagation in headers are on by default.

## Starlette / FastAPI - `draive.starlette`, `draive.fastapi`

```python
from fastapi import APIRouter
from draive.fastapi import ServerContext, application
from draive.opentelemetry import OpenTelemetry

router = APIRouter()

@router.get("/notes/{note_id}")
async def get_note(note_id: UUID) -> Mapping[str, Any]:
    return (await Notes.get(note_id)).to_mapping()  # runs inside the request scope

context = ServerContext(
    notes_service(),
    disposables=(storage_client(),),                # instances, entered once on startup
    observability=OpenTelemetry.observability,      # the callable, not its result
)
app = application(context, routers=(router,), exception_handler=handle_500)
```

- `draive.starlette.application` takes `routes=`, `draive.fastapi.application` takes `routers=`;
  other keyword arguments go to Starlette/FastAPI. `draive.fastapi` re-exports the Starlette pieces.
- Every request runs in its own scope of kind `server` inheriting the context's state; incoming
  trace headers are continued and failures recorded.
- Pass the handler of unhandled errors as `exception_handler`, handlers of specific exceptions as
  `exception_handlers`. Translate domain errors into HTTP responses here, at the edge only.
- State derived from the request (caller, tenant) is added by a middleware passed in `middleware=`
  that wraps the app call in `with ctx.updating(Caller(...)):`.
- Startup and shutdown work is a disposable of the context. One `ServerContext` backs one
  application run: build a new context and application for each test.
- Stream with `StreamResponse(generator, media_type="text/event-stream")`; it keeps the request
  scope open until the last chunk.
- A `State` is not a pydantic type: FastAPI refuses it as a return annotation or body parameter.
  Return `state.to_mapping()` and read input with `State.from_mapping(...)`/`from_json(...)`.

## OpenTelemetry - `draive.opentelemetry`

```python
OpenTelemetry.configure(service="app", version="1.0.0", environment="prod", otlp_endpoint=...)

async with ctx.scope("app", observability=OpenTelemetry.observability()):   # called for a scope
    ...
ServerContext(..., observability=OpenTelemetry.observability)              # uncalled for a server
```

`configure` once per process (guard with `OpenTelemetry.configured()`); a second call is refused.
`ServerContext` needs the callable to continue the trace of each request - an instance records every
request as a separate trace. Code keeps using `ctx.log_*`/`ctx.record_*`; logs, metrics and spans
are correlated.

## Configuration - root package

```python
class Limits(Configuration):
    requests: int = 10

async with ctx.scope("app", ConfigurationRepository.volatile(Limits(requests=5))):
    limits = await Limits.load(required=True)     # absent: None; default= is the fallback
```

With `required=True` an absent configuration is built from its defaults, or raises
`ConfigurationMissing` when it cannot be. A `ConfigurationRepository` state supplies the values:
`.volatile(...)` in memory (tests), `PostgresConfigurationRepository` for a table, or your own
loading function.

## Files - root package

```python
async with Files.access("data/report.json", create=True) as file:
    content = await file.read()
    await file.write(b"...")
```

Async and ready without setup. Created files are owner-only (`mode=0o600`); `exclusive=True` holds
an advisory lock for the block; writes are not atomic.
Symbolic links are refused anywhere in the path - mind `/tmp` and `/var` on macOS. Passing
`Files.access(...)` as a disposable installs `File` in the scope: `await File.read()`.
`Files.traverse(path)` lists a directory. Errors are `FileException`.
