from asyncio import Lock
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from importlib.util import find_spec
from time import monotonic
from types import TracebackType
from typing import Any, final
from uuid import UUID, uuid4

from haiway import ctx
from surrealdb import AsyncWsSurrealConnection
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
from draive.surreal.utils import surreal_value, surreal_variable

__all__ = ("SurrealConnection",)

_EMBEDDED_URL_PREFIXES: tuple[str, ...] = ("mem", "file", "surrealkv")


class SurrealConnection:
    __slots__ = (
        "_connection",
        "_credentials",
        "_database",
        "_embedded",
        "_embedded_lock",
        "_namespace",
        "_url",
    )

    def __init__(
        self,
        *,
        url: str | None = None,
        namespace: str = "surrealdb",
        database: str = "surrealdb",
        user: str | None = None,
        password: str | None = None,
        access: str | None = None,
    ) -> None:
        if url is None:
            # embedded engine is optional (`draive[surrealdb_embedded]`), use it only when present
            if find_spec("surrealdb_embedded") is None:
                raise ValueError(
                    "SurrealDB url is required unless the embedded engine is installed,"
                    " use `draive[surrealdb_embedded]` or provide an explicit url"
                )

            url = "mem://"

        self._url: str = url
        self._embedded: bool = url.startswith(_EMBEDDED_URL_PREFIXES)
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
        self._connection: AsyncWsSurrealConnection  # initialized later
        # The embedded (in-process) engine is not safe under concurrent requests:
        # gathered writes into the same HNSW-indexed table (or a KNN search racing
        # such a write) nondeterministically panic its Rust runtime ("bytes ...
        # advance out of bounds", SIGABRT - verified live), killing the whole
        # process. Every embedded `_send` is therefore serialized through this
        # per-connection lock; remote (ws) connections are unaffected.
        self._embedded_lock: Lock = Lock()

    @property
    def connection(self) -> AsyncWsSurrealConnection:
        assert hasattr(self, "_connection")  # nosec: B101
        return self._connection

    async def _open_connection(self) -> None:
        assert not hasattr(self, "_connection")  # nosec: B101

        connection: AsyncWsSurrealConnection
        if self._url.startswith("ws") or self._url.startswith("wss"):
            connection = AsyncWsSurrealConnection(url=self._url)

        elif self._embedded:
            try:
                from surrealdb import AsyncEmbeddedSurrealConnection

            except ImportError as exc:
                raise ValueError(
                    f"Embedded SurrealDB url {self._url} requires `draive[surrealdb_embedded]`"
                ) from exc

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
        try:
            await self._connection.close()

        finally:
            del self._connection

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

        if self._embedded:
            return _EmbeddedSessionContext(
                connection=self._connection,
                namespace=namespace or self._namespace,
                database=database or self._database,
                credentials=credentials,
                lock=self._embedded_lock,
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
        "_lock",
        "_namespace",
    )

    def __init__(
        self,
        connection: AsyncWsSurrealConnection,
        namespace: str,
        database: str,
        credentials: Mapping[str, str],
        lock: Lock,
    ) -> None:
        self._connection: AsyncWsSurrealConnection = connection
        self._namespace: str = namespace
        self._database: str = database
        self._credentials: Mapping[str, str] = credentials
        self._lock: Lock = lock

    async def __aenter__(self) -> SurrealSession:
        # Serialized like every other embedded `_send` (see `SurrealConnection`'s
        # `_embedded_lock`): the embedded engine panics under concurrent requests.
        async with self._lock:
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
            async with self._lock:
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

            case {"result": [raw_id]}:
                # some servers wrap the transaction identifier within a single element list
                self._transaction_id = UUID(str(raw_id))

            case {"result": {"id": raw_id}} if raw_id is not None:
                self._transaction_id = UUID(str(raw_id))

            case {"result": {"txn": raw_id}} if raw_id is not None:
                self._transaction_id = UUID(str(raw_id))

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
            try:
                await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                    RequestMessage(
                        RequestMethod.CANCEL,
                        session=self._session_id,
                        txn=self._transaction_id,
                    ),
                    process="",  # it is more or less ignored by surreal
                )

            except Exception as exc:
                # the original error is the one to surface, the server drops
                # an unfinished transaction together with its session anyways
                ctx.log_error(
                    "Failed to cancel Surreal transaction",
                    exception=exc,
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
            try:
                await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                    RequestMessage(
                        RequestMethod.DETACH,
                        session=self._session_id,
                    ),
                    process="",  # it is more or less ignored by surreal
                )

            except Exception as exc:
                # the original error is the one to surface
                ctx.log_error(
                    "Failed to detach Surreal session",
                    exception=exc,
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
        try:
            await self._connection._send(  # pyright: ignore[reportPrivateUsage]
                RequestMessage(
                    RequestMethod.DETACH,
                    session=self._session_id,
                ),
                process="",  # it is more or less ignored by surreal
            )

        except Exception as exc:
            if exc_val is None:
                raise

            # do not hide the error the session was left with
            ctx.log_error(
                "Failed to detach Surreal session",
                exception=exc,
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
                        yield {key: surreal_value(element) for key, element in elements.items()}

                    case _:
                        # SELECT VALUE and similar projections can return scalar list items.
                        # Preserve them instead of dropping by wrapping into a single-field object.
                        yield {"value": surreal_value(item)}

        case {"status": "OK", "result": None}:
            return  # no results to produce

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
    params: Mapping[str, Any] = surreal_variable(variables)
    started: float = monotonic()
    response: Mapping[str, Any]
    try:
        response = await connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.QUERY,
                query=statement,
                params=params,
                session=session_id,
                txn=transaction_id,
            )
            if transaction_id
            else RequestMessage(
                RequestMethod.QUERY,
                query=statement,
                params=params,
                session=session_id,
            ),
            bypass=True,
            process="",  # it is more or less ignored by surreal
        )

    except Exception as exc:
        _record_query(started, outcome="error", rows=None)
        raise SurrealException(f"Surreal execution error for: {statement}") from exc

    try:
        rows: Sequence[SurrealObject] = tuple(_process_response(response))

    except BaseException:
        _record_query(started, outcome="error", rows=None)
        raise

    _record_query(started, outcome="success", rows=len(rows))
    return rows


async def _execute_embedded(
    connection: AsyncWsSurrealConnection,
    /,
    statement: str,
    *,
    variables: Mapping[str, SurrealValue],
) -> Sequence[SurrealObject]:
    started: float = monotonic()
    response: Mapping[str, Any]
    try:
        response = await connection._send(  # pyright: ignore[reportPrivateUsage]
            RequestMessage(
                RequestMethod.QUERY,
                query=statement,
                params=surreal_variable(variables),
            ),
            bypass=True,
            process="",  # it is more or less ignored by surreal
        )

    except Exception as exc:
        _record_query(started, outcome="error", rows=None)
        raise SurrealException(f"Surreal execution error for: {statement}") from exc

    try:
        rows: Sequence[SurrealObject] = tuple(_process_response(response))

    except BaseException:
        _record_query(started, outcome="error", rows=None)
        raise

    _record_query(started, outcome="success", rows=len(rows))
    return rows


def _record_query(
    started: float,
    /,
    *,
    outcome: str,
    rows: int | None,
) -> None:
    # statements and variables are never recorded, they may contain user data
    ctx.record_info(
        metric="surreal.query.duration",
        value=(monotonic() - started) * 1000,
        unit="ms",
        kind="histogram",
        attributes={
            "surreal.outcome": outcome,
            "surreal.rows": rows,
        },
    )
