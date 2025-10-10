from collections.abc import Sequence

from haiway import TypeSpecification

from draive import State


def test_validation() -> None:
    class ModelState(State):
        specification: Sequence[TypeSpecification]

    specification: Sequence[TypeSpecification] = (
        {  # union
            "type": ("integer", "string", "null"),
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
        {  # tuple-like array
            "type": "array",
            "prefixItems": ({"type": "integer"},),
            "items": False,
        },
        {  # object
            "type": "object",
            "properties": {
                "integer": {
                    "type": "integer",
                }
            },
            "required": ("integer",),
            "additionalProperties": False,
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
