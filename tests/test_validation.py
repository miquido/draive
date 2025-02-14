from collections.abc import Sequence

from draive import State
from draive.parameters import ParameterSpecification


def test_validation() -> None:
    class ModelState(State):
        specification: Sequence[ParameterSpecification]

    specification: Sequence[ParameterSpecification] = (
        {  # union
            "oneOf": (
                {
                    "type": "integer",
                },
                {
                    "type": "string",
                    "enum": ("enum",),
                },
                {
                    "type": "null",
                },
            ),
        },
        {  # string format
            "type": "string",
            "format": "uuid",
        },
        {  # string enum
            "type": "string",
            "enum": ("enum",),
        },
        {  # string
            "type": "string",
        },
        {  # integer enum
            "type": "integer",
            "enum": (42,),
        },
        {  # integer
            "type": "integer",
        },
        {  # number enum
            "type": "number",
            "enum": (42.0,),
        },
        {  # number
            "type": "number",
        },
        {  # boolean
            "type": "boolean",
        },
        {  # null
            "type": "null",
        },
        {  # array typed
            "type": "array",
            "items": {
                "type": "integer",
            },
        },
        {  # tuple
            "type": "array",
            "prefixItems": (
                {
                    "type": "integer",
                },
            ),
        },
        {  # array any
            "type": "array",
        },
        {  # object
            "type": "object",
            "properties": {
                "integer": {
                    "type": "integer",
                }
            },
        },
        {  # dict
            "type": "object",
            "additionalProperties": {
                "type": "integer",
            },
        },
        {  # any object
            "type": "object",
            "additionalProperties": True,
        },
        {  # reference
            "$ref": "Reference",
        },
    )
    # not raises and has same value
    assert ModelState(specification=specification).specification == specification
