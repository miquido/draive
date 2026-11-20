# Building blocks

Everything a piece of code needs is a `State` found with `ctx.state(Type)`. Pick the kind of
dependency by what it is:

| Kind | What it is | How it is built | How it is installed |
| --- | --- | --- | --- |
| Data | values of the domain | `State` | passed and returned as arguments |
| Configuration | plain settings (bucket, timeout, path) | `State` with defaults, `DefaultEnv` for the environment | instance in `ctx.scope(...)`, or nothing when the defaults do |
| Service | behavior (store a blob, send mail) | `State` holding `Protocol`-typed functions + `@statemethod` | a factory result, e.g. `blobs()` |
| Resource | something to open and close (client, pool, file) | async context manager yielding `State` (a *disposable*) | `disposables=(...)` of the scope or preset |

## Modelling with State

```python
from collections.abc import Sequence
from typing import Annotated, Literal

from draive import Alias, DefaultEnv, Description, Missing, Sensitive, State, Verifier


def positive(value: int) -> int:  # a Verifier raises to reject and returns the value
    if value <= 0:
        raise ValueError("Value has to be positive")
    return value


class Card(State):
    kind: Literal["card"] = "card"
    number: Annotated[str, Sensitive()]


class Transfer(State):
    kind: Literal["transfer"] = "transfer"
    iban: str


type Payment = Card | Transfer  # a union instead of optional fields valid only together


class Invoice(State):
    total_cents: Annotated[int, Verifier(positive)]
    customer: Annotated[str, Alias("customer_id"), Description("External ID")]
    payment: Payment
    lines: Sequence[str] = ()
    note: str | Missing = Missing
    api_key: Annotated[str, Sensitive()] = DefaultEnv("API_KEY", required=True)

    @property
    def total(self) -> float:  # derived, never stored
        return self.total_cents / 100
```

- Construction validates and coerces: `lines=["a"]` is stored as `("a",)`, a `Mapping` as an
  immutable `Map`, a `Set` as `frozenset`. A failure raises `ValidationError` naming the path
  (`Validation of .total_cents failed: ...`); it is not a `ValueError`.
- Change with `invoice.updating(total_cents=...)`, which returns a new instance.
- Serialize with `to_mapping()`/`to_json()`, read with `from_mapping()`/`from_json()`/`validate()`.
  An `Alias` is the name on the wire (`customer_id`) and is accepted by the constructor too;
  `Missing` values are left out of the output. `Sensitive` only redacts `repr`/`str`.
- Describe a type for schemas (structured output, tool arguments) with the class keyword
  `class Invoice(State, description="...")` - stored as `__DESCRIPTION__`, written into the schema
  and kept under `python -OO`, unlike a docstring. A `Description` on the attribute holding it wins.
- A `State` with attributes cannot be subclassed. Share attributes by holding an instance, type a
  variant with a generic (`class Page[T](State)`), share behavior with an attribute-less base.
- States cannot be pickled and the class cannot be patched after creation.
- `case Missing:` captures; use `case draive.Missing:` or `is Missing`.

## Configuration

```python
class StorageConfig(State):
    bucket: str = "default"
    token: Annotated[str, Sensitive()] = DefaultEnv("STORAGE_TOKEN", required=True)
```

Small, immutable, defaulted. A configuration constructible without arguments does not have to be
installed: `ctx.state(StorageConfig)` builds it from its defaults. Override for a block with
`with ctx.updating(StorageConfig(...)):`. `ctx.state(StorageConfig).updating(...)` only returns an
updated copy - pass it explicitly, it does not change what a service call resolves.

## Service (the functionality triple)

```python
# types.py - declarations only
from typing import Protocol, runtime_checkable

@runtime_checkable
class BlobStoring(Protocol):                     # one narrow Protocol per operation
    async def __call__(self, key: str, content: bytes) -> None: ...
```

```python
# state.py
from draive import State, statemethod

from .types import BlobStoring

class Blobs(State):
    store: BlobStoring

    @statemethod
    async def put(self, key: str, content: bytes) -> None:
        await self.store(key, content)
```

```python
# implementation.py
from draive import ctx

from .state import Blobs

async def _store(key: str, content: bytes) -> None:
    config = ctx.state(StorageConfig)            # dependencies resolved at call time
    client = ctx.state(StorageClient)
    ...

def blobs() -> Blobs:                            # factory chooses the implementation
    return Blobs(store=_store)
```

```python
async with ctx.scope("request", blobs()):
    await Blobs.put("key", b"...")               # class call resolves ctx.state(Blobs)
```

- `@statemethod` is written with `self` first; called on the class it uses `ctx.state(cls)`, called
  on an instance it uses that instance.
- Several implementations of one contract (S3, files, in-memory) are several factories returning the
  same `State`; the caller never knows which one it got.
- Keep each Protocol small; a service with many operations holds many fields, not one fat Protocol.
- Implementation functions are module-private; the package exports the `State`, the types and the
  factories. A field may instead default to an implementation, which makes the state available
  without installing it.
- Do not capture dependencies in closures or constructors - resolve them with `ctx.state(...)`
  inside the function, so scopes and tests can replace them.

## Resource (disposable)

```python
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

class StorageClient(State):
    ...

@asynccontextmanager
async def storage_client() -> AsyncGenerator[StorageClient]:
    client = await open_client()
    try:
        yield StorageClient(...)                 # the yielded State joins the scope
    finally:
        await client.close()                     # always runs on scope exit

async with ctx.scope("app", blobs(), disposables=(storage_client(),)):
    ...
```

A disposable yields one `State` or a collection of them. `ctx.scope` and `ServerContext` take
**instances** (`storage_client()`); `ContextPresets` take **factories** (`storage_client`) so each
scope gets a fresh resource. Use `async with ctx.disposables(...)` to add one to the current scope
for a block. Never open clients at import time or keep them in module globals; translate the
client's exceptions into typed ones at this boundary.

## Feature

A feature is a higher level `State` whose methods orchestrate lower-level services through their
`@statemethod`s - never through an integration directly.

```python
class Reports(State):
    @statemethod
    async def publish(self, name: str) -> None:
        await Blobs.put(f"reports/{name}", render(name))   # resolves Blobs from the context
```

An attribute-less `State` like `Reports` needs no installing. A feature with swappable parts follows
the same triple as a service.

## Wiring

Dependencies are assembled once, at the edge (entrypoint, request handler, test):

```python
from draive import ContextPresets

prod = ContextPresets.of(
    "prod",
    blobs(), StorageConfig(bucket="b"),
    disposables=(storage_client,),               # factory
)

async with ctx.scope(prod):                      # or ctx.scope("app", blobs(), disposables=(...))
    async with ctx.scope("job", JobState(...)):  # nested scopes inherit the parent's state
        await Reports.publish("q1")
```

State is keyed by its exact type. When the same type comes from several sources the priority is,
highest first: explicit state, explicit disposables, preset state, parent scope - so
`ctx.scope(prod, Blobs(store=fake))` overrides one piece. Register presets with
`with ctx.presets(prod, dev):` to pick one by name: `ctx.scope("prod")`.

## Tasks

```python
task = ctx.spawn(process, item)                  # joined by this scope on exit
a, b = await gather(fetch_a(), fetch_b())        # all or nothing, the rest cancelled on failure
results = await execute_concurrently(handler, items, concurrent_tasks=8)  # ordered, limited
```

A task a scope waits on must be able to finish by itself. Background work (`ctx.spawn_background`)
keeps logging but has no scope state. See `recipes.md` for choosing among the helpers.

## Errors

```python
class ProviderException(Exception):
    def __init__(self, *, operation: str, identifier: str) -> None:
        super().__init__(f"Provider '{operation}' failed for {identifier}")
        self.operation = operation
        self.identifier = identifier


try:
    await client.call()
except SdkError as exc:
    raise ProviderException(operation="call", identifier=ident) from exc
```

## Tests

```python
import pytest
from draive import ctx


@pytest.mark.asyncio
async def test_publish() -> None:
    stored: dict[str, bytes] = {}

    async def fake_store(key: str, content: bytes) -> None:
        stored[key] = content

    async with ctx.scope("test", Blobs(store=fake_store)):
        await Reports.publish("q")

    assert "reports/q" in stored
```

Fake a dependency by installing a `State` with stub functions; no network, no patching of modules
or globals. A feature is tested with fake solutions, a solution with a fake integration `State`;
only the integration touches the real thing.

## Checklist when adding one

- New behavior: Protocol(s) in `types.py`, `State` + `@statemethod` in `state.py`, functions and a
  factory in `implementation.py`, exports in `__init__.py`.
- New external system: its own integration package, a disposable yielding a client `State`,
  errors translated, secrets `Sensitive()`.
- Features use solutions, solutions wrap integrations; nothing imports an entrypoint.
- Dependencies come from `ctx.state(...)`, are wired at the edge, and have a fake for tests.
