import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Final
from uuid import UUID

from haiway import Pagination
from surrealdb import Duration, RecordID, Table
from surrealdb.cbor import CBORTag
from surrealdb.data.types.constants import TAG_DURATION_COMPACT

from draive.surreal.types import SurrealID, SurrealValue

__all__ = (
    "pagination_offset",
    "surreal_identifier",
    "surreal_value",
    "surreal_variable",
)

_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def pagination_offset(
    pagination: Pagination,
    /,
) -> int:
    if pagination.token is None:
        return 0

    if isinstance(pagination.token, int):
        return max(pagination.token, 0)

    if isinstance(pagination.token, str):
        try:
            return max(int(pagination.token), 0)

        except ValueError as exc:
            raise ValueError("Invalid SurrealDB pagination token") from exc

    raise ValueError("Invalid SurrealDB pagination token")


def surreal_identifier(
    value: str,
    /,
) -> str:
    """Verify a table/relation name before interpolating it into a statement.

    SurrealQL has no parameter form for table names, they have to be inlined,
    therefore each one has to be constrained to safe characters only.
    """
    if _IDENTIFIER_PATTERN.fullmatch(value):
        return value

    raise ValueError(f"Invalid SurrealDB identifier: {value!r}")


def surreal_value(  # noqa: PLR0911
    value: Any,
    /,
) -> SurrealValue:
    """Normalize a value into its canonical draive representation.

    Used when decoding statement results and when normalizing filter operands.
    The wire representation is produced separately by `surreal_variable`.
    """
    if isinstance(
        value,
        UUID | datetime | timedelta | str | int | float | bytes | bytearray | bool | None,
    ):
        return value

    elif isinstance(value, Decimal):
        return float(value)

    elif isinstance(value, RecordID) and isinstance(value.id, UUID | str | int):
        return SurrealID(
            table=value.table_name,
            record=value.id,
        )

    elif isinstance(value, SurrealID):
        return value

    elif isinstance(value, Duration):
        # the SDK decodes durations as an elapsed nanoseconds count
        return timedelta(microseconds=value.elapsed / 1000)

    elif isinstance(value, Table):
        # table names have to be fed in as Table, SurrealDB rejects
        # plain strings in statements like DELETE
        return value

    elif isinstance(value, Mapping):
        return {key: surreal_value(element) for key, element in value.items()}  # pyright: ignore[reportUnknownVariableType]

    elif isinstance(value, Sequence):
        return [surreal_value(element) for element in value]  # pyright: ignore[reportUnknownVariableType]

    else:
        raise ValueError(f"Unsupported Surreal value: {type(value)}")


def surreal_variable(  # noqa: PLR0911
    value: Any,
    /,
) -> Any:
    """Convert a value into the representation SurrealDB accepts on the wire.

    The inverse of `surreal_value` - whatever a statement produces has to be
    usable as a variable again. Applied to every statement variable by the
    connection, so the rest of the integration keeps working on plain values.
    """
    if isinstance(value, SurrealID):
        # SurrealID is what we produce for record links on the way out,
        # convert it back to the SDK representation to allow feeding it in.
        return RecordID(value.table, value.record)

    elif isinstance(value, timedelta):
        # timedelta is what we produce for durations on the way out
        return _duration_tag(round(value / timedelta(microseconds=1)) * 1000)

    elif isinstance(value, Duration):
        return _duration_tag(value.elapsed)

    elif isinstance(value, Decimal):
        return float(value)

    elif isinstance(value, str | bytes | bytearray):
        return value  # strings are sequences, keep them before the Sequence branch

    elif isinstance(value, Mapping):
        return {key: surreal_variable(element) for key, element in value.items()}  # pyright: ignore[reportUnknownVariableType]

    elif isinstance(value, Sequence):
        return [surreal_variable(element) for element in value]  # pyright: ignore[reportUnknownVariableType]

    else:
        # everything else is left to the SDK encoder, including its own types
        return value


def _duration_tag(
    nanoseconds: int,
    /,
) -> CBORTag:
    """Encode a duration under the compact duration tag.

    The SDK encodes its `Duration` as a [seconds, nanoseconds] pair under the
    *string* duration tag instead, which SurrealDB refuses to parse - the embedded
    engine reports a decoding error while a server drops the request without any
    response, leaving the connection waiting for it forever.
    """
    seconds, remainder = divmod(nanoseconds, 1_000_000_000)
    return CBORTag(TAG_DURATION_COMPACT, [seconds, remainder])
