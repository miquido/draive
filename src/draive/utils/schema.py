import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from haiway import TypeSpecification

__all__ = (
    "simplified_schema",
    "strict_schema",
)


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


def _simplified_schema_property(  # noqa: C901, PLR0912, PLR0911
    specification: TypeSpecification,
) -> dict[str, Any] | list[Any] | str:
    match specification:
        case {"$ref": str() as reference}:
            # a reference points back to an anchored, recursive element -
            # naming it is all that can be rendered without recursing forever
            return reference

        case {"type": "null", "description": str() as description}:
            return _described("null", description)

        case {"type": "null"}:
            return "null"

        case {"type": "boolean", "description": str() as description, "enum": [*selection]}:
            return _described(_enum_alternatives(selection), description)

        case {"type": "boolean", "enum": [*selection]}:
            return _enum_alternatives(selection)

        case {"type": "boolean", "description": str() as description}:
            return _described("boolean", description)

        case {"type": "boolean"}:
            return "boolean"

        case {"type": "integer", "description": str() as description, "enum": [*selection]}:
            return _described(_enum_alternatives(selection), description)

        case {"type": "integer", "enum": [*selection]}:
            return _enum_alternatives(selection)

        case {"type": "integer", "description": str() as description}:
            return _described("integer", description)

        case {"type": "integer"}:
            return "integer"

        case {"type": "number", "description": str() as description, "enum": [*selection]}:
            return _described(_enum_alternatives(selection), description)

        case {"type": "number", "enum": [*selection]}:
            return _enum_alternatives(selection)

        case {"type": "number", "description": str() as description}:
            return _described("number", description)

        case {"type": "number"}:
            return "number"

        case {"type": "string", "description": str() as description, "format": format}:
            return _described(format, description)

        case {"type": "string", "format": format}:
            return format

        case {"type": "string", "description": str() as description, "enum": [*selection]}:
            return _described(_enum_alternatives(selection), description)

        case {"type": "string", "enum": [*selection]}:
            return _enum_alternatives(selection)

        case {"type": "string", "description": str() as description}:
            return _described("string", description)

        case {"type": "string"}:
            return "string"

        case {"anyOf": [*alternatives], "description": str() as description}:
            return _described(_type_alternatives(alternatives), description)

        case {"anyOf": [*alternatives]}:
            return _type_alternatives(alternatives)

        case {
            "type": "array",
            "items": False,
            "prefixItems": [*items],
            "description": str() as description,
        }:
            return [
                _simplified_schema_property(
                    specification=item,
                )
                for item in items
            ]  # TODO: add description?

        case {"type": "array", "items": False, "prefixItems": [*items]}:
            return [
                _simplified_schema_property(
                    specification=item,
                )
                for item in items
            ]

        case {"type": "array", "items": items, "description": str() as description}:
            assert items is not False  # nosec: B101
            return [_simplified_schema_property(specification=items)]  # TODO: add description?

        case {"type": "array", "items": items}:
            assert items is not False  # nosec: B101
            return [
                _simplified_schema_property(
                    specification=items,
                ),
            ]

        case {"type": "array", "description": str() as description}:
            return []  # TODO: add description?

        case {"type": "array"}:
            return []

        case {"type": "object", "properties": {**properties}, "description": str() as description}:
            return {
                key: _simplified_schema_property(
                    specification=specification,
                )
                for key, specification in properties.items()
            }  # TODO: add description?

        case {"type": "object", "properties": {**properties}}:
            return {
                key: _simplified_schema_property(
                    specification=specification,
                )
                for key, specification in properties.items()
            }

        case {"type": "object", "additionalProperties": True}:
            return {}

        case {"enum": [*selection], "description": str() as description}:
            return _described(_enum_alternatives(selection), description)

        case {"enum": [*selection]}:
            return _enum_alternatives(selection)

        case {"type": [*alternatives], "description": str() as description}:
            return _described("|".join(alternatives), description)

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


# Strict mode validates string formats against a fixed set and rejects any other one.
# An unsupported format is only a hint for the model, dropping it keeps the schema
# strict which is worth more than the hint.
_SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset(
    (
        "date",
        "date-time",
        "time",
        "uuid",
    )
)
# Objects carrying properties are what counts towards the nesting limit, arrays and
# open maps nested within them are not.
_NESTING_LIMIT: Final[int] = 10
# Types allowed within a `type` alternatives array - an "object" member there would
# still require `additionalProperties` and an "array" member `items`, neither of which
# an alternatives array has a place for.
_ALTERNATIVE_TYPES: Final[frozenset[str]] = frozenset(
    (
        "boolean",
        "integer",
        "null",
        "number",
        "string",
    )
)
# Every keyword a model or tool declaration can produce, except the ones strict mode
# has no equivalent for - `prefixItems`, where a tuple needs `items` of `false`, and
# `$anchor`/`$ref`, where a reference resolves only against a `$defs` entry while a
# recursive type produces an inline anchor. Anything outside of this makes the schema
# untranslatable instead of being passed through into a rejected request.
_ALLOWED_KEYWORDS: Final[frozenset[str]] = frozenset(
    (
        "additionalProperties",
        "anyOf",
        "description",
        "enum",
        "format",
        "items",
        "properties",
        "required",
        "type",
    )
)


class _Unsupported(Exception):
    """Raised for a schema shape which strict mode cannot express."""


def strict_schema(
    schema: Mapping[str, Any],
    /,
    *,
    open_maps: bool = True,
) -> Mapping[str, Any] | None:
    """Convert a json schema into its OpenAI strict mode equivalent.

    Strict mode is the only way the api enforces a schema - without it the schema is
    a hint and the model is free to answer with content which fails decoding, or to
    call a tool with arguments which fail validation. Its additional requirements are
    met by rewriting `required` to hold every property of each object, which makes the
    model always fill a field instead of leaving it to its default.

    Parameters
    ----------
    schema
        Json schema describing a requested output or a tool argument object.
    open_maps
        Whether a mapping of free form keys can be part of the result. Strict mode
        keeps one within a requested output but strips it out of a tool argument
        object, where enforcing the schema would silently drop everything the model
        wanted to put there - a tool argument schema has to be rejected instead.

    Returns
    -------
    Mapping[str, Any] | None
        Strict mode equivalent of the schema, or ``None`` when it holds something
        strict mode has no equivalent for - a tuple, an unbounded array, a value of
        any type, a recursive reference, nesting beyond the supported depth, a keyword
        outside the translated vocabulary, or a mapping which cannot be closed.
    """
    try:
        converted: Mapping[str, Any] = _converted(
            schema,
            nesting=0,
            open_maps=open_maps,
        )

    except _Unsupported:
        return None

    # strict mode requires an object at the root, anything else is rejected
    if converted.get("type") != "object" or "properties" not in converted:
        return None

    return converted


def _converted(
    schema: Mapping[str, Any],
    /,
    *,
    nesting: int,
    open_maps: bool,
) -> Mapping[str, Any]:
    if not schema.keys() <= _ALLOWED_KEYWORDS:
        raise _Unsupported()

    match schema.get("anyOf"):
        case None:
            pass

        case [*alternatives]:
            return {
                **schema,
                "anyOf": [
                    _converted(
                        alternative,
                        nesting=nesting,
                        open_maps=open_maps,
                    )
                    for alternative in alternatives
                ],
            }

        case _:
            raise _Unsupported()

    match schema.get("type"):
        case "object":
            return _converted_object(
                schema,
                nesting=nesting,
                open_maps=open_maps,
            )

        case "array":
            return _converted_array(
                schema,
                nesting=nesting,
                open_maps=open_maps,
            )

        case "string":
            return _converted_string(schema)

        case [*alternatives] if not set(alternatives) <= _ALTERNATIVE_TYPES:
            raise _Unsupported()

        case _:  # scalars, enums and scalar alternatives carry no nested schema
            return schema


def _converted_string(
    schema: Mapping[str, Any],
    /,
) -> Mapping[str, Any]:
    match schema.get("format"):
        case None:
            return schema

        case str() as supported if supported in _SUPPORTED_FORMATS:
            return schema

        case _:
            return {key: value for key, value in schema.items() if key != "format"}


def _converted_object(
    schema: Mapping[str, Any],
    /,
    *,
    nesting: int,
    open_maps: bool,
) -> Mapping[str, Any]:
    nesting += 1
    if nesting > _NESTING_LIMIT:
        raise _Unsupported()

    match schema.get("properties"):
        case None:
            pass

        case {**properties}:
            converted: dict[str, Mapping[str, Any]] = {
                key: _converted(
                    value,
                    nesting=nesting,
                    open_maps=open_maps,
                )
                for key, value in properties.items()
            }

            return {
                **schema,
                "properties": converted,
                # `required` has to hold every property, except the ones strict mode
                # strips out of the schema - an open map, including one behind an
                # array, is rejected within `required` instead of being demanded
                "required": [key for key, value in converted.items() if not _is_open_map(value)],
                "additionalProperties": False,
            }

        case _:
            raise _Unsupported()

    # an object without properties carries its value schema instead
    match schema.get("additionalProperties"):
        case {**values} if open_maps:
            return {
                **schema,
                "additionalProperties": _converted(
                    values,
                    nesting=nesting,
                    open_maps=open_maps,
                ),
            }

        case _:  # `additionalProperties` of `true` cannot be closed, `false` allows nothing
            raise _Unsupported()


def _converted_array(
    schema: Mapping[str, Any],
    /,
    *,
    nesting: int,
    open_maps: bool,
) -> Mapping[str, Any]:
    match schema.get("items"):
        case {**items}:
            return {
                **schema,
                # an array is not a nesting level of its own
                "items": _converted(
                    items,
                    nesting=nesting,
                    open_maps=open_maps,
                ),
            }

        case _:
            raise _Unsupported()  # an array without an item schema is rejected


def _is_open_map(
    schema: Mapping[str, Any],
    /,
) -> bool:
    # an array of open maps is stripped from a strict schema just like a bare one
    while schema.get("type") == "array":
        match schema.get("items"):
            case {**items}:
                schema = items

            case _:
                return False

    return schema.get("type") == "object" and "properties" not in schema
