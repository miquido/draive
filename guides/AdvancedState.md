# Advanced state

Draive provides a dedicated base class for defining application inner state (`State`) and serializable models (`DataModel`). While basic usage is quite straight forward, it hides a lot of customization and features to explore when needed. `State` and `DataModel` share some of the functionalities but have distinct bases and distinct purposes. This notebook focuses on `DataModel`, `State` functionalities are limited to simple validation and immutability.

## Dict conversion

First interesting feature not visible at a glance is an ability to convert `DataModel` from and to a dictionary populated with basic values (dict, list, str, int, float, bool or None).


```python
from draive import BasicValue, DataModel


class DictConverted(DataModel):
    name: str
    value: int


# all of this applies to the `DataModel` as well
dict_converted: DictConverted = DictConverted(name="converted", value=42)
dict_converted_as_dict: dict[str, BasicValue] = dict_converted.as_dict()
dict_converted_from_dict: DictConverted = DictConverted.from_dict(dict_converted_as_dict)
print(dict_converted_from_dict)
```

    name: converted
    value: 42


## Mutations

Both `State` and `DataModel` are immutable by default. Attempting to change any of its fields value will result in both linting and runtime errors. The only valid method to apply a mutation is through a copy. There is a dedicated method to perform a mutating copy operation:


```python
class Mutable(DataModel):
    identifier: str
    value: int


# prepare an instance of state
initial: Mutable = Mutable(
    identifier="pre",
    value=42,
)
# update one of the fields by creating a copy
updated: Mutable = initial.updated(identifier="post")
# update initial state once more - this will be another copy
final: Mutable = initial.updated(value=21)

print("initial", initial)
print("updated", updated)
print("final", final)
```

    initial identifier: pre
    value: 42
    updated identifier: post
    value: 42
    final identifier: pre
    value: 21


## JSON conversion

Each `DataModel` can be serialized using JSON format. Current implementation uses an intermediate step of conversion from/to a dict using the methods described above.


```python
from draive import DataModel


class JSONConverted(DataModel):
    name: str
    value: int


json_converted: JSONConverted = JSONConverted(name="converted", value=42)
json_converted_as_json: str = json_converted.as_json()
json_converted_from_json: JSONConverted = JSONConverted.from_json(json_converted_as_json)
print(json_converted_from_json)
```

    name: converted
    value: 42


## Model schema

Each `DataModel` has an associated schema which is generated using type annotations of the class fields. Models have an ability to generate a JSON-schema compatible description:


```python
class BasicModel(DataModel):
    field: str

print(BasicModel.json_schema(indent=2))
```

    {
      "type": "object",
      "properties": {
        "field": {
          "type": "string"
        }
      },
      "required": [
        "field"
      ]
    }


Additionally each model can generate a simplified schema description. This can be useful when requesting LLM structured data generation in some cases. Let's have a look:


```python
print(BasicModel.simplified_schema(indent=2))
```

    {
      "field": "string"
    }


Field schema can be altered by using per field customization if needed.

## Field customization

Each field defined within the `DataModel` class can be customized by using a dedicated default value called `Field`. It allows to change the field validation, serialization or other elements.

### Alias and description

Fields can have aliases which can be useful for serialization. Both regular and aliased names are available to instantiate an object. When generating the specification, aliases are always used instead of regular field names. Besides the alias, each field can also have a description which is also included in the schema:


```python
from draive import Field


class CustomizedSchemaModel(DataModel):
    described: int = Field(description="Field description")
    aliased: str = Field(aliased="field_alias")


print(f"JSON schema:\n{CustomizedSchemaModel.json_schema(indent=2)}")
print(f"Simplified schema:\n{CustomizedSchemaModel.simplified_schema(indent=2)}")
```

    JSON schema:
    {
      "type": "object",
      "properties": {
        "described": {
          "type": "integer",
          "description": "Field description"
        },
        "field_alias": {
          "type": "string"
        }
      },
      "required": [
        "described",
        "field_alias"
      ]
    }
    Simplified schema:
    {
      "described": "integer(Field description)",
      "field_alias": "string"
    }


### Default values

Fields can have regular default values instead of a dedicated `Field` default. When using `Field` you can still provide a default value and you can also define default value factory instead. When default value is not defined through the `Field` the `Field` itself does not serve a role of a default value.


```python
from collections.abc import Sequence


class CustomizedDefaultsModel(DataModel):
    default: int = 42
    field_default: int = Field(default=21)
    field_default_factory: Sequence[str] = Field(default_factory=list)


# since all fields have defaults we can initialize without arguments
print(CustomizedDefaultsModel())
```

    default: 42
    field_default: 21
    field_default_factory:


### Verification

Each field is automatically validated based on its type annotation. However if a field is required to have any additional validation you can provide an appropriate function to execute in addition to regular validation.


```python
# verifier gets pre-validated value, it already have required type
def verifier(value: int) -> None:
    if value < 0:
        # raise an Exception if something is wrong with the value
        raise ValueError("Value can't be less than zero!")


class VerifiedModel(DataModel):
    value: int = Field(verifier=verifier)
```

### Validation

When automatically generated validation is not suitable for your specific case you can override it to provide custom validation function for each field. When that happens you are taking the full responsibility of ensuring proper value is used for a given field. You can't specify both validator and verifier at the same time, add any verification required to your validator when needed.


```python
from typing import Any

from draive import ParameterValidationContext, ParameterValidationError


def validator(
    # validator gets unknown type as an input, make sure to verify or convert it
    value: Any,
    /,
    *,
    # validator has also the current validation context which contains additional information
    # i.e. what field actually is validated, it is very useful for preparing diagnostics information
    context: ParameterValidationContext,
) -> int:
    if isinstance(value, int):
        return value

    else:
        raise ParameterValidationError(f"Expected int but received {type(value)}")


class ValidatedModel(DataModel):
    value: int = Field(validator=validator)
```

### Schema specification

Similarly to validation, field schema specification is also automatically generated and can also be replaced with a custom schema specification for any given field.


```python
class CustomizedSpecificationModel(DataModel):
    value: int = Field(specification={"type": "integer", "description": "Fully custom"})


print(CustomizedSpecificationModel.json_schema(indent=2))
```

    {
      "type": "object",
      "properties": {
        "value": {
          "type": "integer",
          "description": "Fully custom"
        }
      },
      "required": [
        "value"
      ]
    }


### Conversion

The last element which can be specified for a field is a function converting field value to the basic type. It is used for the dict conversion or serialization of data. You can fully customize what will be the representation of the field when converting the object to dict or json. Make sure that validating (deserializing) the result back will not cause any issues. You should also ensure proper schema update if needed.


```python
def converter(value: str, /,) -> int:
    return len(value)


class CustomizedConversionModel(DataModel):
    value: str = Field(converter=converter)


print(CustomizedConversionModel(value="integer?"))
```

    value: 8


## Property paths

AttributePath is an additional advanced feature of `DataModel` (also `State`). AttributePath is an object that points to a given element inside the model/state object and can be used to retrieve it when needed. To create an attribute path, you can use the special type property `_`.


```python
from typing import cast

from draive import AttributePath


class NestedPathModel(DataModel):
    values: Sequence[int]


class PathModel(DataModel):
    nested: NestedPathModel
    value: int


# we can construct the path for any given field inside
path: AttributePath[PathModel, Sequence[int]] = cast(
    AttributePath[PathModel, Sequence[int]],
    PathModel._.nested.values,
)
path_model_instance: PathModel = PathModel(
    value=21,
    nested=NestedPathModel(
        values=[42, 21],
    ),
)
# and use it to retrieve that field value from any instance
print(path(path_model_instance))
```

    (42, 21)


Property paths can be used not only as the getters for field values. It also preserves the path as a string to be accessed later if needed.


```python
print(path)
```

    nested.values


Besides that paths can be also used to prepare per field requirements. We can simplify usage of paths in that case by avoiding the type conversion step:


```python
from draive import AttributeRequirement

# prepare a parameter requirement - choose a requirement type you need
requirement: AttributeRequirement[PathModel] = AttributeRequirement[PathModel].equal(
    42, # here we require value to be equal to 42
    path=PathModel._.nested.values[0], # under this specific path
)

# requirement can be executed to check value on any instance
requirement.check(path_model_instance)
```
    True



Requirements can be combined and examined. This can be used to provide an expressive interface for defining various filters.


```python
print("lhs:", requirement.lhs)
print("operator:", requirement.operator)
print("rhs:", requirement.rhs)

combined_requirement: AttributeRequirement[PathModel] = requirement & AttributeRequirement[
    PathModel
].contained_in(
    [12, 21],
    path=PathModel._.value,
)

combined_requirement.check(path_model_instance)
```
    lhs: nested.values[0]
    operator: equal
    rhs: 42

    True


