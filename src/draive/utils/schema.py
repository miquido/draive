import json
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta
from typing import Any, Final

from haiway import State, TypeSpecification

__all__ = ("simplified_schema",)


def simplified_schema(
    specification: TypeSpecification,
    indent: int | None = None,
) -> str:
    match specification:
        case {"properties": {**properties}}:
            return json.dumps(
                {
                    key: _simplified_schema_property(
                        specification=specification,
                    )
                    for key, specification in properties.items()
                },
                indent=indent,
            )

        case other:
            raise ValueError(f"Unsupported basic specification: {other}")


def _simplified_schema_property(  # noqa: PLR0911
    specification: TypeSpecification,
) -> dict[str, Any] | list[Any] | str:
    match specification:
        case {"$ref": str() as reference}:
            # a reference points back to an anchored, recursive element -
            # naming it is all that can be rendered without recursing forever
            return reference

        # structural elements render as json, their descriptions are not representable
        case {"type": "array", "items": False, "prefixItems": [*items]}:
            return [_simplified_schema_property(item) for item in items]

        case {"type": "array", "items": {**item}}:
            return [_simplified_schema_property(item)]  # pyright: ignore[reportArgumentType]

        case {"type": "array"}:
            return []

        case {"type": "object", "properties": {**properties}}:
            return {key: _simplified_schema_property(value) for key, value in properties.items()}

        case {"type": "object", "additionalProperties": True}:
            return {}

        case {"type": "object", "additionalProperties": {**values}}:
            return {"<key>": _simplified_schema_property(values)}  # pyright: ignore[reportArgumentType]

        case {"description": str() as description}:
            return _described(_simplified_inline(specification), description)

        case _:
            return _simplified_inline(specification)


def _simplified_inline(
    specification: TypeSpecification,
) -> str:
    match specification:
        case {"anyOf": [*alternatives]}:
            return _type_alternatives(alternatives)

        case {"type": "string", "format": str() as format}:
            return format

        case {"enum": [*selection]}:
            return _enum_alternatives(selection)

        case {"type": "string", "pattern": str() as pattern} if pattern in _WRITTEN_FORMATS:
            return _WRITTEN_FORMATS[pattern]

        case {"type": "null" | "boolean" | "integer" | "number" | "string" as simple}:
            return simple

        case {"type": [*alternatives]}:
            return "|".join(alternatives)

        case other:
            raise ValueError(f"Unsupported basic specification element: {other}")


def _described(
    rendered: str,
    description: str,
    /,
) -> str:
    return f"{rendered}({description})" if description else rendered


def _type_alternatives(
    alternatives: Sequence[Any],
    /,
) -> str:
    elements: list[str] = []
    for alternative in alternatives:
        match _simplified_schema_property(specification=alternative):
            case str() as rendered:
                elements.append(rendered)

            case list() | dict() as rendered:
                elements.append(json.dumps(rendered))

    return "|".join(elements)


def _enum_alternatives(
    selection: Sequence[Any],
    /,
) -> str:
    return "|".join(_enum_element(element) for element in selection)


def _enum_element(
    value: Any,
    /,
) -> str:
    match value:
        case str() as text:
            return f"'{text}'"

        case bool() as boolean:
            return "true" if boolean else "false"

        case None:
            return "null"

        case other:
            return str(other)


def _written_formats() -> Mapping[str, str]:
    # haiway states the spelling of these values as a pattern, no format fits it - the patterns
    # are taken from haiway itself and named for the model by what they spell
    class _Written(State):
        moment: datetime
        clock: time
        duration: timedelta

    properties: Mapping[str, Any] = json.loads(_Written.json_schema())["properties"]
    names: Mapping[str, str] = {"moment": "date-time", "clock": "time", "duration": "duration"}
    # a spelling haiway states differently is rendered as its plain type instead
    return {
        pattern: name
        for key, name in names.items()
        if isinstance(pattern := properties.get(key, {}).get("pattern"), str)
    }


_WRITTEN_FORMATS: Final[Mapping[str, str]] = _written_formats()
