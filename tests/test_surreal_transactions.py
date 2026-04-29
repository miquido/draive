from uuid import uuid4

import pytest
from haiway import State
from surrealdb.request_message.methods import RequestMethod

from draive.surreal import SurrealClient, SurrealException
from draive.surreal import connection as surreal_connection
from draive.surreal.types import SurrealObject


class _SurrealTransactionItem(State):
    value: int


@pytest.mark.asyncio
async def test_surreal_client_session_uses_stubbed_ws_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[RequestMethod, dict[str, object]]] = []

    class _FakeConnection:
        def __init__(self, url: str) -> None:
            self.url = url

        async def connect(self, url: str | None = None) -> None:
            self.url = url or self.url

        async def _send(
            self,
            message: object,
            process: str,
            bypass: bool = False,
        ) -> dict[str, object]:
            _ = process, bypass
            method = message.method
            kwargs = message.kwargs
            calls.append((method, kwargs))
            if method is RequestMethod.QUERY:
                return {"result": [{"status": "OK", "result": [{"value": 7}]}]}

            return {"result": None}

        async def close(self) -> None:
            return None

    monkeypatch.setattr(surreal_connection, "AsyncWsSurrealConnection", _FakeConnection)

    async with SurrealClient(
        url="ws://example.test",
        namespace="default_ns",
        database="default_db",
        user="root",
        password="root",
    ) as client:
        async with client.prepare_session(namespace="custom_ns", database="custom_db") as session:
            rows: list[SurrealObject] = list(await session.execute("RETURN $value;", value=7))

    assert rows == [{"value": 7}]
    assert [method for method, _ in calls] == [
        RequestMethod.ATTACH,
        RequestMethod.USE,
        RequestMethod.SIGN_IN,
        RequestMethod.QUERY,
        RequestMethod.DETACH,
    ]
    assert calls[1][1]["namespace"] == "custom_ns"
    assert calls[1][1]["database"] == "custom_db"
    assert calls[2][1]["params"] == {"user": "root", "password": "root"}
    assert calls[3][1]["params"] == {"value": 7}


@pytest.mark.asyncio
async def test_surreal_client_transaction_uses_transaction_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_txn: object | None = None
    commit_txn: object | None = None
    transaction_id = uuid4()

    class _FakeConnection:
        def __init__(self, url: str) -> None:
            self.url = url

        async def connect(self, url: str | None = None) -> None:
            self.url = url or self.url

        async def _send(
            self,
            message: object,
            process: str,
            bypass: bool = False,
        ) -> dict[str, object]:
            nonlocal query_txn, commit_txn
            _ = process, bypass
            method = message.method
            kwargs = message.kwargs

            if method is RequestMethod.BEGIN:
                return {"result": str(transaction_id)}

            if method is RequestMethod.QUERY:
                query_txn = kwargs.get("txn")
                return {"result": [{"status": "OK", "result": []}]}

            if method is RequestMethod.COMMIT:
                commit_txn = kwargs.get("txn")
                return {"result": None}

            return {"result": None}

        async def close(self) -> None:
            return None

    monkeypatch.setattr(surreal_connection, "AsyncWsSurrealConnection", _FakeConnection)

    async with SurrealClient(
        url="ws://example.test",
        namespace="ns",
        database="db",
    ) as client:
        async with client.prepare_session() as session:
            async with session.transaction() as transaction_session:
                await transaction_session.execute("RETURN [];")

    assert query_txn == transaction_id
    assert commit_txn == transaction_id


@pytest.mark.asyncio
async def test_surreal_session_transaction_is_unavailable_for_embedded_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEmbeddedConnection:
        def __init__(self, url: str) -> None:
            self.url = url

        async def connect(self, url: str | None = None) -> None:
            self.url = url or self.url

        async def _send(
            self,
            message: object,
            process: str,
            bypass: bool = False,
        ) -> dict[str, object]:
            _ = process, bypass, message
            return {"result": None}

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        surreal_connection, "AsyncEmbeddedSurrealConnection", _FakeEmbeddedConnection
    )

    async with SurrealClient(
        url="mem://",
        namespace="test_surreal_transactions",
        database="embedded_transaction_api",
    ) as client:
        async with client.prepare_session() as session:
            with pytest.raises(
                RuntimeError,
                match="Embedded SurrealDB connections do not support client-side transactions",
            ):
                session.transaction()


@pytest.mark.asyncio
async def test_surreal_embedded_execute_supports_inline_sql_transactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEmbeddedConnection:
        def __init__(self, url: str) -> None:
            self.url = url

        async def connect(self, url: str | None = None) -> None:
            self.url = url or self.url

        async def _send(
            self,
            message: object,
            process: str,
            bypass: bool = False,
        ) -> dict[str, object]:
            _ = process, bypass
            if message.method is RequestMethod.QUERY and "BEGIN TRANSACTION;" in str(
                message.kwargs.get("query", "")
            ):
                detail: str = "Transactions are not supported for embedded connection in this mode"
                return {
                    "result": [
                        {
                            "status": "ERR",
                            "detail": detail,
                        }
                    ]
                }
            return {"result": [{"status": "OK", "result": []}]}

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        surreal_connection, "AsyncEmbeddedSurrealConnection", _FakeEmbeddedConnection
    )

    async with SurrealClient(
        url="mem://",
        namespace="test_surreal_transactions",
        database="embedded_inline_transactions",
    ) as client:
        async with client.prepare_session() as session:
            with pytest.raises(SurrealException):
                await session.execute(
                    "BEGIN TRANSACTION;"
                    " CREATE item:rolled_back CONTENT { value: 1 };"
                    " CANCEL TRANSACTION;"
                )


@pytest.mark.asyncio
async def test_surreal_session_create_and_delete_use_surreal_record_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[RequestMethod, dict[str, object]]] = []

    class _FakeEmbeddedConnection:
        def __init__(self, url: str) -> None:
            self.url = url

        async def connect(self, url: str | None = None) -> None:
            self.url = url or self.url

        async def _send(
            self,
            message: object,
            process: str,
            bypass: bool = False,
        ) -> dict[str, object]:
            _ = process, bypass
            calls.append((message.method, message.kwargs))
            return {"result": [{"status": "OK", "result": []}]}

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        surreal_connection, "AsyncEmbeddedSurrealConnection", _FakeEmbeddedConnection
    )

    async with SurrealClient(
        url="mem://",
        namespace="test_surreal_transactions",
        database="embedded_record_ids",
    ) as client:
        async with client.prepare_session() as session:
            assert await session.create(_SurrealTransactionItem(value=3), identifier="created") == (
                "created"
            )
            await session.delete(_SurrealTransactionItem, "created")

    query_calls: list[dict[str, object]] = [
        kwargs for method, kwargs in calls if method is RequestMethod.QUERY
    ]
    assert len(query_calls) == 2
    assert str(query_calls[0]["params"]["_record"]) == "_SurrealTransactionItem:created"
    assert str(query_calls[1]["params"]["_record"]) == "_SurrealTransactionItem:created"
