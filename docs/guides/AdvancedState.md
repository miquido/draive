# Advanced State

Draive models are typed `State` classes. Schema and JSON support is derived from the declared
field types. Adding `State, serializable=True` turns that into a requirement, so a class whose
fields cannot produce a JSON schema fails at declaration time instead of silently losing it.

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

`strict_schema(...)` sits next to it and converts a JSON schema into its OpenAI strict mode
equivalent, returning `None` when the schema holds something strict mode cannot express.

## Immutability And Updates

```python
from draive import State


class MutableExample(State):
    identifier: str
    value: int


initial = MutableExample(identifier="pre", value=42)
updated = initial.updating(identifier="post")
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
