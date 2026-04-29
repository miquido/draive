from collections.abc import Mapping, Sequence
from datetime import datetime
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable
from uuid import UUID

from haiway import State
from surrealdb import RecordID

if TYPE_CHECKING:
    from draive.surreal.state import SurrealSession

__all__ = (
    "SurrealBasicValue",
    "SurrealException",
    "SurrealID",
    "SurrealObject",
    "SurrealSessionContext",
    "SurrealSessionPreparing",
    "SurrealStatementExecuting",
    "SurrealTransactionContext",
    "SurrealTransactionPreparing",
    "SurrealValue",
)


class SurrealException(Exception):
    pass


@final
class SurrealID(State):
    table: str
    record: UUID | str


SurrealBasicValue = (
    SurrealID | RecordID | UUID | datetime | str | int | float | bytes | bytearray | bool | None
)
type SurrealValue = Mapping[str, SurrealValue] | Sequence[SurrealValue] | SurrealBasicValue
SurrealObject = Mapping[str, SurrealValue]


@runtime_checkable
class SurrealStatementExecuting(Protocol):
    async def __call__(
        self,
        statement: str,
        *,
        variables: Mapping[str, SurrealValue],
    ) -> Sequence[SurrealObject]: ...


@runtime_checkable
class SurrealSessionContext(Protocol):
    async def __aenter__(self) -> SurrealSession: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class SurrealSessionPreparing(Protocol):
    def __call__(
        self,
        namespace: str | None,
        database: str | None,
        user: str | None,
        password: str | None,
        access: str | None,
    ) -> SurrealSessionContext: ...


@runtime_checkable
class SurrealTransactionContext(Protocol):
    async def __aenter__(self) -> SurrealSession: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class SurrealTransactionPreparing(Protocol):
    def __call__(self) -> SurrealTransactionContext: ...
