from collections.abc import Generator, Mapping, MutableMapping, Sequence
from types import TracebackType
from typing import Any, final
from uuid import UUID, uuid4

from surrealdb import AsyncEmbeddedSurrealConnection, AsyncWsSurrealConnection
from surrealdb.errors import parse_query_error, parse_rpc_error
from surrealdb.request_message.message import RequestMessage
from surrealdb.request_message.methods import RequestMethod

from draive.surreal.state import SurrealSession
from draive.surreal.types import (
    SurrealException,
    SurrealObject,
    SurrealSessionContext,
    SurrealTransactionContext,
    SurrealValue,
)
from draive.surreal.utils import surreal_value

__all__ = ("SurrealConnection",)


class SurrealConnection:
    __slots__ = (
        "_connection",
        "_credentials",
        "_database",
        "_namespace",
        "_url",
    )

    def __init__(
        self,
        *,
        url: str = "mem://",
        namespace: str = "surrealdb",
        database: str = "surrealdb",
        user: str | None = None,
        password: str | None = None,
        access: str | None = None,
    ) -> None:
        self._url: str = url
        credentials: MutableMapping[str, str] = {}

        if user:
            credentials["user"] = user

        if password:
            credentials["password"] = password

        if access:
            credentials["access"] = access

        self._namespace: str = namespace
        self._database: str = database
        self._credentials: Mapping[str, str] = credentials
        self._connection: AsyncWsSurrealConnection | None = None

    @property
    def connection(self) -> AsyncWsSurrealConnection:
        assert self._connection is not None  # nosec: B101
        return self._connection

    async def _open_connection(self) -> None:
        if self._connection is not None:
            await self._close_connection()

        connection: AsyncWsSurrealConnection
        if self._url.startswith("ws") or self._url.startswith("wss"):
            connection = AsyncWsSurrealConnection(url=self._url)

        elif (
            self._url.startswith("mem")
            or self._url.startswith("file")
            or self._url.startswith("surrealkv")
        ):
            connection = AsyncEmbeddedSurrealConnection(url=self._url)

        else:
            raise ValueError(f"Unsupported SurrealDB url: {self._url}")

        try:
            await connection.connect(self._url)

        except BaseException:
            await connection.close()
            raise  # reraise

        self._connection = connection

    async def _close_connection(self) -> None:
        if self._connection is None:
            return  # no connection

        try:
            await self._connection.close()

        finally:
            self._connection = None

    def prepare_session(
        self,
        namespace: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        access: str | None = None,
    ) -> SurrealSessionContext:
        assert self._connection is not None  # nosec: B101
        credentials: MutableMapping[str, str] = dict(self._credentials)
        if user:
            credentials["user"] = user

        if password:
            credentials["password"] = password

        if access:
            credentials["access"] = access

        if isinstance(self._connection, AsyncEmbeddedSurrealConnection):
            return _EmbeddedSessionContext(
                connection=self._connection,
                namespace=namespace or self._namespace,
                database=database or self._database,
                credentials=credentials,
            )

        else:
            return _SessionContext(
                connection=self._connection,
                namespace=namespace or self._namespace,
                database=database or self._database,
                credentials=credentials,
                session_id=uuid4(),
            )


@final
class _EmbeddedSessionContext:
    __slots__ = (
        "_connection",
        "_credentials",
        "_database",
        "_namespace",
    )

    def __init__(
        self,
        connection: AsyncEmbeddedSurrealConnection,
        namespace: str,
        database: str,
        credentials: Mapping[str, str],
    ) -> None:
        self._connection: AsyncEmbeddedSurrealConnection = connection
        self._namespace: str = namespace
        self._database: str = database
        self._credentials: Mapping[str, str] = credentials

    async def __aenter__(self) -> SurrealSession:
        await self._connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.USE,
                namespace=self._namespace,
                database=self._database,
            ),
            process="",  # it is more or less ignored by surreal
        )

        if self._credentials:
            await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                RequestMessage(
                    RequestMethod.SIGN_IN,
                    params=self._credentials,
                ),
                process="",  # it is more or less ignored by surreal
            )

        async def execute(
            statement: str,
            *,
            variables: Mapping[str, SurrealValue],
        ) -> Sequence[SurrealObject]:
            return await _execute_embedded(
                self._connection,
                statement,
                variables=variables,
            )

        def transaction() -> SurrealTransactionContext:
            raise RuntimeError(
                "Embedded SurrealDB connections do not support client-side transactions"
            )

        return SurrealSession(
            statement_executing=execute,
            transaction_preparing=transaction,
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


@final
class _TransactionContext:
    __slots__ = (
        "_connection",
        "_session_id",
        "_transaction_id",
    )

    def __init__(
        self,
        connection: AsyncWsSurrealConnection,
        session_id: UUID,
    ) -> None:
        self._connection: AsyncWsSurrealConnection = connection
        self._session_id: UUID = session_id
        self._transaction_id: UUID  # created later

    async def __aenter__(self) -> SurrealSession:
        match await self._connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.BEGIN,
                session=self._session_id,
            ),
            process="",  # it is more or less ignored by surreal
        ):
            case {"result": str() as raw_id}:
                self._transaction_id = UUID(raw_id)

            case {"result": UUID() as id}:
                self._transaction_id = id

            case _:
                raise SurrealException("Invalid Surreal response")

        async def execute(
            statement: str,
            *,
            variables: Mapping[str, SurrealValue],
        ) -> Sequence[SurrealObject]:
            return await _execute(
                self._connection,
                statement,
                session_id=self._session_id,
                transaction_id=self._transaction_id,
                variables=variables,
            )

        def transaction() -> SurrealTransactionContext:
            raise RuntimeError("Nested transactions are not supported")

        return SurrealSession(
            statement_executing=execute,
            transaction_preparing=transaction,
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is None:
            await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                RequestMessage(
                    RequestMethod.COMMIT,
                    session=self._session_id,
                    txn=self._transaction_id,
                ),
                process="",  # it is more or less ignored by surreal
            )

        else:
            await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                RequestMessage(
                    RequestMethod.CANCEL,
                    session=self._session_id,
                    txn=self._transaction_id,
                ),
                process="",  # it is more or less ignored by surreal
            )


@final
class _SessionContext:
    __slots__ = (
        "_connection",
        "_credentials",
        "_database",
        "_namespace",
        "_session_id",
    )

    def __init__(
        self,
        connection: AsyncWsSurrealConnection,
        namespace: str,
        database: str,
        credentials: Mapping[str, str],
        session_id: UUID,
    ) -> None:
        self._connection: AsyncWsSurrealConnection = connection
        self._namespace: str = namespace
        self._database: str = database
        self._credentials: Mapping[str, str] = credentials
        self._session_id: UUID = session_id

    async def __aenter__(self) -> SurrealSession:
        await self._connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.ATTACH,
                session=self._session_id,
            ),
            process="",  # it is more or less ignored by surreal
        )

        try:
            await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                RequestMessage(
                    RequestMethod.USE,
                    namespace=self._namespace,
                    database=self._database,
                    session=self._session_id,
                ),
                process="",  # it is more or less ignored by surreal
            )

            if self._credentials:
                await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                    RequestMessage(
                        RequestMethod.SIGN_IN,
                        params=self._credentials,
                        session=self._session_id,
                    ),
                    process="",  # it is more or less ignored by surreal
                )

        except BaseException:
            await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                RequestMessage(
                    RequestMethod.DETACH,
                    session=self._session_id,
                ),
                process="",  # it is more or less ignored by surreal
            )
            raise

        async def execute(
            statement: str,
            *,
            variables: Mapping[str, SurrealValue],
        ) -> Sequence[SurrealObject]:
            return await _execute(
                self._connection,
                statement,
                session_id=self._session_id,
                transaction_id=None,
                variables=variables,
            )

        def transaction() -> SurrealTransactionContext:
            return _TransactionContext(
                connection=self._connection,
                session_id=self._session_id,
            )

        return SurrealSession(
            statement_executing=execute,
            transaction_preparing=transaction,
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.DETACH,
                session=self._session_id,
            ),
            process="",  # it is more or less ignored by surreal
        )


def _process_statement_result(  # noqa: C901, PLR0912
    result: Any,
) -> Generator[SurrealObject]:
    match result:
        case {"status": "OK", "result": {**content}}:
            yield {key: surreal_value(element) for key, element in content.items()}

        case {"status": "OK", "result": [*items]}:
            for item in items:
                match item:
                    case None:
                        continue

                    case {**elements}:
                        yield {key: surreal_value(element) for key, element in elements.items()}  # pyright: ignore[reportUnknownVariableType]

                    case _:
                        # SELECT VALUE and similar projections can return scalar list items.
                        # Preserve them instead of dropping by wrapping into a single-field object.
                        yield {"value": surreal_value(item)}

        case {"status": "OK", "result": None}:
            return  # no results to produce

        case {"status": "OK", "time": str(), "result": _}:
            return  # statement summary with scalar result, not an object row

        case {"status": "OK", "result": result}:
            # Some statements may return scalar values directly.
            yield {"value": surreal_value(result)}

        case {"status": "OK"}:
            return  # no results to produce

        case {"status": "ERR"}:
            cause: BaseException = parse_query_error(result)
            raise SurrealException(f"Surreal execution error: {cause}") from cause

        case {"status": str() as status, "result": detail}:
            raise SurrealException(f"Invalid Surreal result status: {status}, {detail}")

        case {"status": str() as status}:
            raise SurrealException(f"Invalid Surreal result status: {status}")

        case _:
            raise SurrealException("Invalid Surreal response")


def _process_response(
    response: Mapping[str, Any],
) -> Generator[SurrealObject]:
    match response:
        case {"error": {**error}}:
            cause: BaseException = parse_rpc_error(error)
            raise SurrealException(f"Surreal execution error: {cause}") from cause

        case {"error": str() as error}:
            raise SurrealException(f"Surreal execution error: {error}")

        case {"result": [*results]}:
            for result in results:
                yield from _process_statement_result(result)

        case {"result": result}:
            yield from _process_statement_result(result)

        case _:
            raise SurrealException("Invalid Surreal response")


async def _execute(
    connection: AsyncWsSurrealConnection,
    /,
    statement: str,
    *,
    session_id: UUID,
    transaction_id: UUID | None,
    variables: Mapping[str, SurrealValue],
) -> Sequence[SurrealObject]:
    response: Mapping[str, Any]
    try:
        response = await connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.QUERY,
                query=statement,
                params=variables,
                session=session_id,
                txn=transaction_id,
            )
            if transaction_id
            else RequestMessage(
                RequestMethod.QUERY,
                query=statement,
                params=variables,
                session=session_id,
            ),
            bypass=True,
            process="",  # it is more or less ignored by surreal
        )

    except Exception as exc:
        raise SurrealException(f"Surreal execution error for: {statement}") from exc

    return tuple(_process_response(response))


async def _execute_embedded(
    connection: AsyncEmbeddedSurrealConnection,
    /,
    statement: str,
    *,
    variables: Mapping[str, SurrealValue],
) -> Sequence[SurrealObject]:
    response: Mapping[str, Any]
    try:
        response = await connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.QUERY,
                query=statement,
                params=variables,
            ),
            bypass=True,
            process="",  # it is more or less ignored by surreal
        )

    except Exception as exc:
        raise SurrealException(f"Surreal execution error for: {statement}") from exc

    return tuple(_process_response(response))
