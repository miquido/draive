from collections.abc import Sequence
from typing import Any, Literal

from haiway import State
from haiway.attributes.annotations import (
    BoolAttribute,
    BytesAttribute,
    DateAttribute,
    DatetimeAttribute,
    FloatAttribute,
    IntegerAttribute,
    IntEnumAttribute,
    LiteralAttribute,
    MappingAttribute,
    MetaAttribute,
    MissingAttribute,
    NoneAttribute,
    ObjectAttribute,
    OpenTupleAttribute,
    PathAttribute,
    SequenceAttribute,
    SetAttribute,
    StrEnumAttribute,
    StringAttribute,
    TimeAttribute,
    TimedeltaAttribute,
    TupleAttribute,
    TypedDictAttribute,
    UnionAttribute,
    UUIDAttribute,
    ValidableAttribute,
)

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
    statements: list[str] = []
    # types come from the attributes as haiway resolved them - the JSON schema of a State accepts
    # the written forms of a value and states no format of a datetime, it tells too little apart
    for field in model.__FIELDS__:
        surreal_type, flexible = _surreal_type(field.annotation)
        statements.append(
            # FLEXIBLE has to follow the type, a SurrealDB server rejects
            # the reversed order accepted by older versions with a parse error
            f"DEFINE FIELD IF NOT EXISTS {surreal_field_path((field.key,))} ON TABLE {table_name} "
            f"TYPE {surreal_type}{' FLEXIBLE' if flexible else ''};"
        )

    return tuple(statements)


_SCALAR_TYPES: tuple[tuple[tuple[type[Any], ...], str], ...] = (
    ((DatetimeAttribute,), "datetime"),
    ((BoolAttribute,), "bool"),
    ((IntegerAttribute, IntEnumAttribute), "int"),
    ((FloatAttribute,), "float"),
    (
        (
            StringAttribute,
            StrEnumAttribute,
            UUIDAttribute,
            PathAttribute,
            DateAttribute,
            TimeAttribute,
            TimedeltaAttribute,
            BytesAttribute,
        ),
        "string",
    ),
)


def _surreal_type(  # noqa: PLR0911
    annotation: Any,
    /,
) -> tuple[str, bool]:
    for attribute_types, surreal_type in _SCALAR_TYPES:
        if isinstance(annotation, attribute_types):
            return surreal_type, False

    match annotation:
        case ValidableAttribute(attribute=validated):
            return _surreal_type(validated)

        case UnionAttribute(alternatives=alternatives):
            return _surreal_union_type(alternatives)

        case SequenceAttribute(values=values) | SetAttribute(values=values):
            item_type, item_flexible = _surreal_type(values)
            return f"array<{item_type}>", item_flexible

        case TupleAttribute() | OpenTupleAttribute():
            return "array", False

        case ObjectAttribute() | MappingAttribute() | TypedDictAttribute() | MetaAttribute():
            return "object", True

        case LiteralAttribute(values=values):
            return _surreal_literal_type(values), False

        case NoneAttribute() | MissingAttribute():
            return "option<any>", False

        case _:
            return "any", False


def _surreal_union_type(
    alternatives: Sequence[Any],
    /,
) -> tuple[str, bool]:
    # `None` and `Missing` both leave the field without a value - a missing one is left out of the
    # record, which a field of a plain type refuses
    present: Sequence[Any] = [
        alternative
        for alternative in alternatives
        if not isinstance(alternative, NoneAttribute | MissingAttribute)
    ]
    surreal_type: str
    flexible: bool
    if len(present) == 1:
        surreal_type, flexible = _surreal_type(present[0])

    else:
        surreal_type, flexible = "any", False

    if len(present) != len(alternatives) and not surreal_type.startswith("option<"):
        return f"option<{surreal_type}>", flexible

    return surreal_type, flexible


def _surreal_literal_type(
    values: Sequence[Any],
    /,
) -> str:
    if all(isinstance(value, bool) for value in values):
        return "bool"

    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "int"

    if all(isinstance(value, str) for value in values):
        return "string"

    return "any"
