from collections.abc import Awaitable, Callable, Mapping, Sequence

import pytest

from draive.surreal.state import SurrealSession
from draive.surreal.types import SurrealObject, SurrealTransactionContext, SurrealValue

__all__ = ("surreal_session",)


def surreal_session(
    execute: Callable[..., Awaitable[Sequence[SurrealObject]]],
    /,
) -> SurrealSession:
    async def executing(
        statement: str,
        /,
        *,
        variables: Mapping[str, SurrealValue],
    ) -> Sequence[SurrealObject]:
        return await execute(statement, **variables)

    def preparing() -> SurrealTransactionContext:
        pytest.fail("unexpected transaction")

    return SurrealSession(
        statement_executing=executing,
        transaction_preparing=preparing,
    )
