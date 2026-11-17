# Style, immutability and functional principles

## Design principles in practice

The three principles of `SKILL.md` apply to every decision, new code and review. Model alternatives
as a union of small types, closed `Literal`s or enums and narrow with `match`/`isinstance`; derive
with a `@property` or a function instead of storing; import a name, value or list from its home
instead of repeating it.

```python
# bad: `error` and `value` only valid in some states
class Result(State):
    ok: bool
    value: str | None = None
    error: str | None = None

# good: each state carries exactly what it needs
class Success(State):
    value: str

class Failure(State):
    error: str

type Result = Success | Failure
```

## Immutability

- Data is a `State`; it cannot be mutated, assigned to or deleted from. "Change" is
  `value.updating(field=...)`, which returns a new instance. Write transformations as functions or
  methods returning new values, never in-place updates.
- Public types use abstract immutable protocols - `Sequence`, `Mapping`, `Set` - which `State`
  stores as `tuple`, immutable `Map` and `frozenset`. Never annotate `list`/`dict`/`set`; never keep
  a mutable module-level container as shared state.
- A `State` class itself is immutable: do not assign class attributes after creation, patch it, or
  add attributes with decorators. A `State` declaring attributes is final and has a single base -
  compose by holding instances, parametrize a generic, or inherit from an attribute-less `State` to
  share behavior; no mixins or multiple inheritance. States cannot be pickled: use
  `to_json`/`to_mapping`.
- A small record that must hold what it is given untouched (callables, clients) is a `@final` class
  with `__slots__` that binds values once with `object.__setattr__` and whose `__setattr__` and
  `__delattr__` raise (typed `NoReturn`). An object whose internals legitimately change (a client,
  a response caching its body) is a plain `__slots__` class - do not pretend it is immutable.
- Mutable state that must exist lives in the context (a scoped `State`, a disposable's resource),
  not in globals or singletons.
- Where a value crosses a context boundary (spawned task, queue, event), assert that it is
  immutable/sendable and keep the message free of secrets.

## Typing

- Strictest typing syntax of Python 3.15: `type` aliases, PEP 695 generics (`class Box[T](State)`,
  `def first[T](items: Sequence[T]) -> T`), `X | None`, `Self`, `TypeIs`, `Literal`, `Final`.
- Every public function, method and attribute is fully annotated; code passes `pyright` strict
  with no ignores. No untyped public API, no loose `Any` except at third-party boundaries, no `cast`
  unless provably safe and local.
- Prefer explicit attribute access to `getattr`/`setattr`; use dynamic access only at narrow
  boundaries.
- Use `@final` on classes not designed for inheritance; prefer composition and Protocols
  (structural typing) for clients and adapters; `@runtime_checkable` when checked at runtime.
- Positional-only (`/`) and keyword-only (`*`) markers make call sites unambiguous; one parameter
  per line once a signature wraps, with trailing commas.
- Name Protocols as gerunds describing the operation (`UserFetching`, `NoteCreating`), `State`s
  after what they hold or provide, factories after what they build.

## Functional style

- Prefer small pure functions and data in, data out; push side effects (I/O, time, randomness) to
  the edges and behind Protocols so they can be replaced.
- Behavior is injected as functions held in `State`, not subclassed. Avoid inheritance hierarchies,
  abstract base classes with state, and manager/singleton objects.
- No hidden dependencies: a function's inputs are its arguments plus what it explicitly resolves
  with `ctx.state(...)`.
- `match` over `if/elif` chains on type or shape; early returns over deep nesting.

## Async

- All I/O is `async`; never call blocking code in a coroutine. Bridge unavoidable sync APIs with
  `@asynchronous`. No custom threads or raw `asyncio.create_task`/`asyncio.gather` - use
  `ctx.spawn`, `gather`, `concurrently`.
- Await long-running operations directly; keep boundaries async. Every spawned task has an owner
  that awaits it: the scope, or the code holding the task.

## Errors

- Raise typed exceptions, never bare `Exception`; translate third-party exceptions at the boundary
  with actionable context (provider, operation, identifiers) and `raise ... from exc`.
- Misuse of an API is a programming error - `assert` it; expected runtime failures are typed
  exceptions. Failure messages help debugging but never leak secrets.

## Logging and observability

- `ctx.log_*` for messages, `ctx.record_*` for attributes, events and metrics; never `print` or
  `logging`. Surface user-facing errors as structured events before raising. A record carries an
  event or a metric, never both.
- Never log secrets, tokens or whole request bodies; mark secrets `Sensitive()`.

## Documentation

- Public symbols get NumPy-style docstrings: summary, Parameters, Returns, Raises, and the
  rationale where it is not obvious; add an example that uses `ctx.scope` when it helps.
- Private helpers: no docstrings, self-explanatory names. No module docstrings. Comments explain
  *why*, not *what*.
- When behavior or the public API changes, update docs and examples and the changelog.
