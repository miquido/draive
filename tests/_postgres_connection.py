from collections.abc import Awaitable, Callable, Sequence
from typing import cast

import pytest
from haiway.postgres import (
    Postgres,
    PostgresConnection,
    PostgresRow,
    PostgresTransactionIsolation,
    PostgresValue,
)
from haiway.postgres.types import PostgresConnectionContext, PostgresTransactionContext

__all__ = ("postgres_states",)


def postgres_states(
    *,
    fetch: Callable[..., Awaitable[object]] | None = None,
    fetch_one: Callable[..., Awaitable[object]] | None = None,
    execute: Callable[..., Awaitable[object]] | None = None,
) -> tuple[Postgres, PostgresConnection]:
    """Postgres and connection states serving the given fakes, `fetch_one` yields a row or none."""

    async def fetching(
        statement: str,
        /,
        *args: PostgresValue,
    ) -> Sequence[PostgresRow]:
        if fetch is not None:
            return cast(Sequence[PostgresRow], await fetch(statement, *args))

        if fetch_one is not None:
            row = await fetch_one(statement, *args)
            return () if row is None else (cast(PostgresRow, row),)

        pytest.fail("unexpected fetch")

    async def executing(
        statement: str,
        /,
        *args: PostgresValue,
    ) -> str:
        if execute is None:
            pytest.fail("unexpected execute")

        await execute(statement, *args)
        return "OK"

    def transaction(
        *,
        isolation: PostgresTransactionIsolation | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> PostgresTransactionContext:
        pytest.fail("unexpected transaction")

    def acquiring() -> PostgresConnectionContext:
        pytest.fail("unexpected connection acquiring")

    return (
        Postgres(connection_acquiring=acquiring),
        PostgresConnection(
            statement_fetching=fetching,
            statement_executing=executing,
            transaction_preparing=transaction,
        ),
    )
