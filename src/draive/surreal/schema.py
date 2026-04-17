from collections.abc import Mapping, Sequence
from json import loads
from typing import Any, Literal, cast

from haiway import State

__all__ = (
    "SurrealTableKind",
    "surreal_field_definitions",
)

type SurrealTableKind = Literal["ANY", "NORMAL", "RELATION"]


def surreal_field_definitions[Model: State](
    model: type[Model],
    /,
    *,
    table: str | None = None,
) -> Sequence[str]:
    table_name: str = table or model.__name__
    schema: Mapping[str, Any] = cast(Mapping[str, Any], loads(model.json_schema()))
    properties: Mapping[str, Any] = cast(Mapping[str, Any], schema.get("properties") or {})

    statements: list[str] = []
    for field_name, field_schema in properties.items():
        surreal_type, flexible = _surreal_field_type(cast(Mapping[str, Any], field_schema))
        statements.append(
            f"DEFINE FIELD IF NOT EXISTS {field_name} ON TABLE {table_name} "
            f"{'FLEXIBLE ' if flexible else ''}TYPE {surreal_type};"
        )

    return tuple(statements)


def _surreal_field_type(
    schema: Mapping[str, Any],
    /,
) -> tuple[str, bool]:
    if "oneOf" in schema:
        return _surreal_union_type(cast(Sequence[Mapping[str, Any]], schema["oneOf"]))

    if "anyOf" in schema:
        return _surreal_union_type(cast(Sequence[Mapping[str, Any]], schema["anyOf"]))

    schema_type: object = schema.get("type")
    if isinstance(schema_type, Sequence) and not isinstance(schema_type, str):
        variants: list[Mapping[str, Any]] = []
        for value in cast(Sequence[object], schema_type):
            if isinstance(value, str):
                variants.append({"type": value})

        return _surreal_union_type(variants)

    if not isinstance(schema_type, str):
        return "any", False

    return _surreal_simple_type(
        schema_type,
        value_format=cast(str | None, schema.get("format")),
        items=cast(Mapping[str, Any] | None, schema.get("items")),
    )


def _surreal_union_type(
    variants: Sequence[Mapping[str, Any]],
    /,
) -> tuple[str, bool]:
    non_null: list[Mapping[str, Any]] = [
        variant for variant in variants if variant.get("type") != "null"
    ]
    nullable: bool = len(non_null) != len(variants)

    if len(non_null) == 1:
        surreal_type, flexible = _surreal_field_type(non_null[0])
        if nullable:
            return f"option<{surreal_type}>", flexible

        return surreal_type, flexible

    if nullable:
        return "option<any>", False

    return "any", False


def _surreal_simple_type(
    schema_type: str,
    /,
    *,
    value_format: str | None,
    items: Mapping[str, Any] | None,
) -> tuple[str, bool]:
    if schema_type == "array":
        return _surreal_array_type(items)

    if schema_type == "string" and value_format == "date-time":
        return "datetime", False

    simple_types: Mapping[str, tuple[str, bool]] = {
        "string": ("string", False),
        "integer": ("int", False),
        "number": ("float", False),
        "boolean": ("bool", False),
        "object": ("object", True),
        "null": ("option<any>", False),
    }
    return simple_types.get(schema_type, ("any", False))


def _surreal_array_type(
    items: Mapping[str, Any] | None,
    /,
) -> tuple[str, bool]:
    if items is None:
        return "array", False

    item_type, item_flexible = _surreal_field_type(items)
    return f"array<{item_type}>", item_flexible
