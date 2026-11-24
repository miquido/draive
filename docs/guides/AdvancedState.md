# Advanced State

Draive models are typed `State` classes. Schema and JSON support is derived from the declared field
types. Adding `State, serializable=True` turns that into a requirement, so a class whose fields
cannot produce a JSON schema fails at declaration time instead of silently losing it.

## Serialization And Schema

```python
from draive import State
from draive.utils import simplified_schema


class Example(State, serializable=True):
    name: str
    value: int


item = Example(name="converted", value=42)
encoded = item.to_json()
decoded = Example.from_json(encoded)

print(decoded)
print(Example.json_schema(indent=2))
print(simplified_schema(Example.__SPECIFICATION__, indent=2))
```

`json_schema(...)` is precise and machine-oriented; `simplified_schema(...)` is compact and often
useful when injecting schema hints into LLM prompts.

## Decoding External Input

Decoding is strict: a key which is neither an attribute name nor an alias is refused, and so is a
key a `TypedDict` attribute does not declare (`Unexpected key 'y' for Point`), in every way a
`State` is built. When the input comes from a source which may add fields of its own, such as a
third party API, pass `ignore_extra=True` to `from_mapping`, `from_json` or `from_json_array`.
Unknown keys are then skipped in every nested `State`, `TypedDict`, sequence, mapping and union the
decoding reaches.

```python
from typing import TypedDict

from draive import State


class Point(TypedDict):
    x: int


class Item(State):
    name: str
    point: Point


payload = '{"name": "a", "point": {"x": 1, "y": 2}, "added_later": true}'

item = Item.from_json(payload, ignore_extra=True)  # Item(name="a", point={"x": 1})
Item.from_json(payload)  # raises ValueError, the keys are unknown
```

The constructor, `updating(...)` and `validate(...)` keep refusing unknown keys, only the decoding
helpers take the flag. In a union the first alternative which validates wins once keys are skipped,
so list the most specific alternative first or tell them apart with a `Literal` attribute. A
`TypedDict | Mapping[...]` attribute given undeclared keys selects the mapping, keeping every key.

## Immutability And Updates

```python
from draive import State


class MutableExample(State):
    identifier: str
    value: int


initial = MutableExample(identifier="pre", value=42)
updated = initial.updating(identifier="post")
```

## Optional Values With Missing

`Missing` is a sentinel which is both the value and the type, the way `None` is. Use it for
attributes which may be left unset, and compare with `is`.

```python
from draive import Missing, State


class Options(State):
    temperature: float | Missing = Missing
    seed: int | Missing = Missing


options = Options(temperature=0.2)
seed = None if options.seed is Missing else options.seed
```

`value is Missing` and `value is not Missing` narrow the type, so no helper is needed. `Missing` is
truthy, so `if value:` does not tell it apart. Within `match`, use a dotted name
(`case haiway.Missing:`), because a bare `case Missing:` captures the value.

## State Class Rules

- A `State` class has a single base, the `State` it refines, which declares no attributes. Share
    behavior through an attribute-less base and compose by holding instances as attributes.
- A `State` class cannot be changed once it is created. Declare methods and class variables in the
    class body. Assigning an attribute to the class afterwards, including a shared instance such as
    `Model.empty = Model(...)`, raises `TypeError`; create such a value lazily instead.
- Function attributes are checked when the state is created: a value which cannot be called the way
    the annotation says is refused, instead of failing once it is called.

## Testing With State

Because State classes cannot be patched, give a test the fakes through the state itself and run it
inside a scope, instead of replacing methods with `mock.patch.object` or `monkeypatch`.

```python
from draive import ctx

async with ctx.scope("test", GenerativeModel(generating=fake_generating)):
    ...
```

## Schema Customization With Annotations

You can enrich field schema using `Annotated` metadata.

```python
from typing import Annotated

from draive import Alias, Description, Specification, State


class Customized(State, serializable=True):
    described: Annotated[int, Description("Field description")]
    aliased: Annotated[str, Alias("field_alias")]
    fully_custom: Annotated[int, Specification({"type": "integer", "description": "Custom"})]


print(Customized.json_schema(indent=2))
```

Describe the type itself with the `description` class keyword. It is written into the schema of the
type, also when the type is nested in another `State` (structured generation, tool arguments), and
it is kept under `python -OO`, unlike a docstring. A `Description` on the attribute holding the type
takes precedence, and a specialization of a generic keeps the description of its origin.

```python
from draive import State


class AssistantConfig(State, serializable=True, description="Settings of the assistant"):
    model: str = "claude-sonnet-5-5"


AssistantConfig.__DESCRIPTION__  # "Settings of the assistant"
```

## Defaults And Factories

```python
from collections.abc import Sequence

from draive import Default, State


class Defaults(State, serializable=True):
    retries: int = 3
    tags: Sequence[str] = Default(factory=tuple)
```

## Attribute Paths And Requirements

Attribute paths let you reference fields in a typed way for indexing, filtering, and reusable rules.

```python
from collections.abc import Sequence
from typing import cast

from draive import AttributePath, AttributeRequirement, State


class Nested(State, serializable=True):
    values: Sequence[int]


class Model(State, serializable=True):
    nested: Nested
    value: int


instance = Model(nested=Nested(values=(42, 21)), value=21)

# `Model._` mimics the model type for static typing, so paths pass straight into
# APIs accepting them - like the requirement below or `VectorIndex.index(attribute=...)`.
# Keeping one around and applying it yourself requires a local cast to its actual type.
path = cast(AttributePath[Model, Sequence[int]], Model._.nested.values)
print(path(instance))

req = AttributeRequirement[Model].equal(42, path=Model._.nested.values[0])
print(req.check(instance))
```
