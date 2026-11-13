from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from haiway import ctx
from haiway.postgres import Postgres, PostgresConnection

__all__ = ("postgres_transaction",)


@asynccontextmanager
async def postgres_transaction() -> AsyncGenerator[PostgresConnection]:
    # reuse the connection bound to the current scope (nested transactions become
    # savepoints), acquiring a new one when there is none - acquiring within
    # a bound connection is forbidden
    if ctx.contains_state(PostgresConnection):
        connection: PostgresConnection = ctx.state(PostgresConnection)
        async with connection.transaction():
            yield connection

    else:
        async with Postgres.acquire_connection() as acquired:
            async with acquired.transaction():
                yield acquired
