from base64 import b64decode, b64encode
from typing import Any

from haiway import MISSING, Missing, not_missing

from draive.parameters import (
    Field,
    ParameterValidationContext,
    ParameterValidationError,
)

__all__ = [
    "b64_or_url_field",
]


def b64_or_url_field(
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
        validator=_b64_or_url_validator,
        converter=_b64_or_url_converter,
    )


def _b64_or_url_validator(
    value: Any,
    context: ParameterValidationContext,
) -> str | bytes:
    match value:
        case str() as string:
            if string.startswith("http"):
                return string  # it is url if it starts from http

            try:
                # try decoding base64...
                return b64decode(string)

            except Exception:
                # ...or use as string (url) if it fails
                return string

        case bytes() as data:
            return data

        case _:
            raise ParameterValidationError.invalid_type(
                expected=str | bytes,
                received=value,
                context=context,
            )


def _b64_or_url_converter(
    value: str | bytes,
    /,
) -> str:
    match value:
        case str() as string:
            return string

        case bytes() as data:
            return b64encode(data).decode("utf-8")
