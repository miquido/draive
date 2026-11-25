import types
from collections.abc import Mapping, Sequence
from datetime import datetime
from json import loads
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from haiway import Missing, State

from draive.surreal.utils import surreal_field_path, surreal_identifier

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
    table_name: str = surreal_identifier(table or model.__name__)
    schema: Mapping[str, Any] = cast(Mapping[str, Any], loads(model.json_schema()))
    properties: Mapping[str, Any] = cast(Mapping[str, Any], schema.get("properties") or {})
    # the schema states no format of a datetime, which is written without an offset, and accepts
    # the written forms of a bool - their types are taken from the annotation of the attribute
    hints: Mapping[str, Any] = get_type_hints(model)
    annotations: Mapping[str, Any] = {
        field.key: hints.get(field.name) for field in model.__FIELDS__
    }

    statements: list[str] = []
    for field_name, field_schema in properties.items():
        surreal_type, flexible = _surreal_field_type(
            cast(Mapping[str, Any], field_schema),
            annotations.get(field_name),
        )
        statements.append(
            # FLEXIBLE has to follow the type, a SurrealDB server rejects
            # the reversed order accepted by older versions with a parse error
            f"DEFINE FIELD IF NOT EXISTS {surreal_field_path((field_name,))} ON TABLE {table_name} "
            f"TYPE {surreal_type}{' FLEXIBLE' if flexible else ''};"
        )

    return tuple(statements)


def _surreal_field_type(
    schema: Mapping[str, Any],
    annotation: Any,
    /,
) -> tuple[str, bool]:
    if _omittable(annotation):
        # a missing value is left out of the record, which a field of a plain type refuses
        surreal_type, flexible = _surreal_field_type(schema, _present(annotation))
        if surreal_type.startswith("option<"):
            return surreal_type, flexible

        return f"option<{surreal_type}>", flexible

    # haiway spells unions as `anyOf`
    if "anyOf" in schema:
        return _surreal_union_type(
            cast(Sequence[Mapping[str, Any]], schema["anyOf"]),
            annotation,
        )

    schema_type: object = schema.get("type")
    if isinstance(schema_type, Sequence) and not isinstance(schema_type, str):
        variants: list[Mapping[str, Any]] = []
        for value in cast(Sequence[object], schema_type):
            if isinstance(value, str):
                variants.append({"type": value})

        return _surreal_union_type(variants, annotation)

    if not isinstance(schema_type, str):
        return "any", False

    return _surreal_simple_type(
        schema_type,
        annotation,
        items=cast(Mapping[str, Any] | None, schema.get("items")),
    )


def _surreal_union_type(
    variants: Sequence[Mapping[str, Any]],
    annotation: Any,
    /,
) -> tuple[str, bool]:
    non_null: list[Mapping[str, Any]] = [
        variant for variant in variants if variant.get("type") != "null"
    ]
    surreal_type: str
    flexible: bool
    if _present(annotation) is bool:
        # the schema of a bool accepts its written forms as well, `0`, `1`, `true` and `false`
        surreal_type, flexible = "bool", False

    elif len(non_null) == 1:
        surreal_type, flexible = _surreal_field_type(non_null[0], _present(annotation))

    else:
        surreal_type, flexible = "any", False

    if len(non_null) != len(variants):  # nullable
        return f"option<{surreal_type}>", flexible

    return surreal_type, flexible


def _surreal_simple_type(
    schema_type: str,
    annotation: Any,
    /,
    *,
    items: Mapping[str, Any] | None,
) -> tuple[str, bool]:
    if schema_type == "array":
        return _surreal_array_type(items, _item(annotation))

    if schema_type == "string" and annotation is datetime:
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
    annotation: Any,
    /,
) -> tuple[str, bool]:
    if items is None:
        return "array", False

    item_type, item_flexible = _surreal_field_type(items, annotation)
    return f"array<{item_type}>", item_flexible


def _omittable(
    annotation: Any,
    /,
) -> bool:
    return get_origin(annotation) in (Union, types.UnionType) and any(
        variant is Missing for variant in get_args(annotation)
    )


def _present(
    annotation: Any,
    /,
) -> Any:
    # the annotation of the only variant of an optional value - `None` and `Missing` are left out
    # of a union, a union of more variants is not followed
    if get_origin(annotation) not in (Union, types.UnionType):
        return annotation

    variants: Sequence[Any] = [
        variant
        for variant in get_args(annotation)
        if variant is not type(None) and variant is not Missing
    ]
    return variants[0] if len(variants) == 1 else None


def _item(
    annotation: Any,
    /,
) -> Any:
    # the annotation of the items of a homogeneous sequence or set
    arguments: Sequence[Any] = get_args(annotation)
    return arguments[0] if len(arguments) == 1 else None
