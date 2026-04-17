from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from haiway import Pagination
from surrealdb import Datetime, RecordID

from draive.surreal.types import SurrealID, SurrealValue

__all__ = (
    "pagination_offset",
    "surreal_value",
)


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


def surreal_value(
    value: Any,
    /,
) -> SurrealValue:
    if isinstance(value, UUID | datetime | str | int | float | bytes | bytearray | bool | None):
        return value

    elif isinstance(value, Decimal):
        return float(value)

    elif isinstance(value, RecordID) and isinstance(value.id, UUID | str):
        return SurrealID(
            table=value.table_name,
            record=value.id,
        )

    elif isinstance(value, Datetime):
        return datetime.fromisoformat(value.dt)

    elif isinstance(value, Mapping):
        return {key: surreal_value(element) for key, element in value.items()}  # pyright: ignore[reportUnknownVariableType]

    elif isinstance(value, Sequence):
        return [surreal_value(element) for element in value]  # pyright: ignore[reportUnknownVariableType]

    else:
        raise ValueError(f"Unsupported Surreal value: {type(value)}")
