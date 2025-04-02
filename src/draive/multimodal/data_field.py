from base64 import b64decode, b64encode
from typing import Any

from haiway import MISSING, Missing, not_missing

from draive.parameters import (
    Field,
    ParameterValidationContext,
)

__all__ = ("b64_data_field",)


def b64_data_field(
    *,
    description: str | Missing = MISSING,
) -> Any:
    return Field(
        specification={
            "type": "string",
            "description": description,
        }
        if not_missing(description)
        else {
            "type": "string",
        },
        validator=_b64_validator,
        converter=_b64_converter,
    )


def _b64_validator(
    value: Any,
    context: ParameterValidationContext,
) -> bytes:
    match value:
        case bytes() as data:
            return data

        case str() as string:
            return b64decode(string)

        case _:
            raise TypeError(f"Expected 'str | bytes', received '{type(value).__name__}'")


def _b64_converter(
    value: bytes,
    /,
) -> str:
    return b64encode(value).decode("utf-8")
