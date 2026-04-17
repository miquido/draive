from collections.abc import Mapping, Sequence
from datetime import datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import pytest
from haiway import AttributeRequirement, Pagination, State, ctx
from surrealdb import RecordID

from draive.surreal.state import Surreal, SurrealSession
from draive.surreal.types import SurrealID, SurrealObject


class _ExampleModel(State):
    name: str
    updated: int


class _Profile(State):
    handle: str
    active: bool


class _SchemaModel(State):
    name: str
    score: float
    updated: datetime | None = None
    tags: tuple[str, ...]
    profile: _Profile | None = None


class _Follows(State):
    since: datetime
    strength: int | None = None


def _row(
    *,
    name: str,
    updated: int,
) -> SurrealObject:
    return cast(
        SurrealObject,
        {
            "name": name,
            "updated": updated,
        },
    )


@pytest.mark.asyncio
async def test_surrealdb_fetch_uses_surreal_filter_and_overfetches_for_pagination() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        assert variables == {
            "_f0": "alpha",
            "_limit": 3,
            "_start": 0,
        }
        assert "WHERE name = $_f0" in statement
        return (
            _row(name="alpha", updated=3),
            _row(name="alpha", updated=2),
            _row(name="alpha", updated=1),
        )

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    page = await database.fetch(
        _ExampleModel,
        pagination=Pagination.of(limit=2),
        requirements=AttributeRequirement.equal("alpha", _ExampleModel._.name),
    )

    assert tuple(item.name for item in page.items) == ("alpha", "alpha")
    assert page.pagination.token == 2
    assert len(execution_calls) == 1


@pytest.mark.asyncio
async def test_surrealdb_fetch_does_not_emit_next_token_for_exact_last_page() -> None:
    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        assert "WHERE" not in statement
        assert "ORDER BY" not in statement
        assert variables == {
            "_limit": 3,
            "_start": 0,
        }
        return (
            _row(name="alpha", updated=2),
            _row(name="beta", updated=1),
        )

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    page = await database.fetch(
        _ExampleModel,
        pagination=Pagination.of(limit=2),
    )

    assert tuple(item.name for item in page.items) == ("alpha", "beta")
    assert page.pagination.token is None


@pytest.mark.asyncio
async def test_surrealdb_fetch_uses_provided_order_attribute() -> None:
    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        assert "ORDER BY name DESC, id DESC" in statement
        assert variables == {
            "_limit": 3,
            "_start": 0,
        }
        return (
            _row(name="beta", updated=2),
            _row(name="alpha", updated=1),
        )

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    page = await database.fetch(
        _ExampleModel,
        pagination=Pagination.of(limit=2),
        order=_ExampleModel._.name,
    )

    assert tuple(item.name for item in page.items) == ("beta", "alpha")
    assert page.pagination.token is None


@pytest.mark.asyncio
async def test_surrealdb_delete_with_requirements_uses_surreal_filter_clause() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        return ()

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    await database.delete(
        _ExampleModel,
        requirements=AttributeRequirement.equal("alpha", _ExampleModel._.name),
    )

    assert execution_calls == [
        (
            "DELETE $_table WHERE name = $_f0;",
            {
                "_table": "_ExampleModel",
                "_f0": "alpha",
            },
        )
    ]


@pytest.mark.asyncio
async def test_surrealdb_define_table_supports_state_schema() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        return ()

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    await database.define_table(_SchemaModel, schema=True)

    assert execution_calls == [
        (
            "DEFINE TABLE IF NOT EXISTS _SchemaModel SCHEMAFULL TYPE NORMAL;",
            {},
        ),
        (
            "DEFINE FIELD IF NOT EXISTS name ON TABLE _SchemaModel TYPE string;",
            {},
        ),
        (
            "DEFINE FIELD IF NOT EXISTS score ON TABLE _SchemaModel TYPE float;",
            {},
        ),
        (
            "DEFINE FIELD IF NOT EXISTS updated ON TABLE _SchemaModel TYPE option<datetime>;",
            {},
        ),
        (
            "DEFINE FIELD IF NOT EXISTS tags ON TABLE _SchemaModel TYPE array<string>;",
            {},
        ),
        (
            "DEFINE FIELD IF NOT EXISTS profile ON TABLE _SchemaModel "
            "FLEXIBLE TYPE option<object>;",
            {},
        ),
    ]


@pytest.mark.asyncio
async def test_surrealdb_define_table_accepts_raw_table_identifier() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        return ()

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    await database.define_table("bad table")

    assert execution_calls == [
        ("DEFINE TABLE IF NOT EXISTS bad table SCHEMALESS TYPE NORMAL;", {}),
    ]


@pytest.mark.asyncio
async def test_surrealdb_define_relation_supports_state_schema_without_in_out_fields() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        return ()

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    await database.define_relation(
        _Follows,
        from_table=_ExampleModel,
        to_table=_SchemaModel,
        schema=True,
    )

    assert execution_calls == [
        (
            "DEFINE TABLE IF NOT EXISTS _Follows SCHEMAFULL "
            "TYPE RELATION FROM _ExampleModel TO _SchemaModel;",
            {},
        ),
        (
            "DEFINE FIELD IF NOT EXISTS since ON TABLE _Follows TYPE datetime;",
            {},
        ),
        (
            "DEFINE FIELD IF NOT EXISTS strength ON TABLE _Follows TYPE option<int>;",
            {},
        ),
    ]


@pytest.mark.asyncio
async def test_surrealdb_relate_uses_record_variables_and_content() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        return (cast(SurrealObject, {"id": SurrealID(table="_Follows", record="edge-1")}),)

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    edge_id = await database.relate(
        _ExampleModel,
        "source",
        _Follows,
        _SchemaModel,
        UUID("00000000-0000-0000-0000-000000000001"),
        content=_Follows(
            since=datetime(2026, 1, 1),
            strength=3,
        ),
    )

    assert edge_id == "edge-1"
    assert len(execution_calls) == 1
    statement, variables = execution_calls[0]
    assert statement == "RELATE $_source->_Follows->$_target CONTENT $_content RETURN id;"
    assert variables is not None
    source = variables["_source"]
    assert isinstance(source, RecordID)
    assert source.table_name == "_ExampleModel"
    assert source.id == "source"
    target = variables["_target"]
    assert isinstance(target, RecordID)
    assert target.table_name == "_SchemaModel"
    assert target.id == "00000000000000000000000000000001"
    assert variables["_content"] == {
        "since": datetime(2026, 1, 1),
        "strength": 3,
    }


@pytest.mark.asyncio
async def test_surrealdb_related_fetches_directional_targets_with_pagination() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        return (
            cast(SurrealObject, {"value": {"name": "alpha", "updated": 3}}),
            cast(SurrealObject, {"value": {"name": "beta", "updated": 2}}),
            cast(SurrealObject, {"value": {"name": "gamma", "updated": 1}}),
        )

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    page = await database.related(
        _SchemaModel,
        "source",
        _Follows,
        _ExampleModel,
        direction="in",
        pagination=Pagination.of(limit=2),
    )

    assert tuple(item.name for item in page.items) == ("alpha", "beta")
    assert page.pagination.token == 2
    assert len(execution_calls) == 1
    statement, variables = execution_calls[0]
    assert (
        statement == "SELECT in.* AS value FROM _Follows WHERE out = $_source "
        "ORDER BY id DESC LIMIT $_limit START AT $_start;"
    )
    assert variables is not None
    assert variables["_limit"] == 3
    assert variables["_start"] == 0
    source = variables["_source"]
    assert isinstance(source, RecordID)
    assert source.table_name == "_SchemaModel"
    assert source.id == "source"


@pytest.mark.asyncio
async def test_surreal_create_uses_active_session() -> None:
    execution_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        execution_calls.append((statement, variables))
        return ()

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    async with ctx.scope("test.surreal.active_session", database):
        assert (
            await Surreal.create(
                _ExampleModel(name="alpha", updated=1),
                identifier="example",
            )
            == "example"
        )

    assert len(execution_calls) == 1
    statement, variables = execution_calls[0]
    assert statement == "CREATE $_record CONTENT $_content;"
    assert variables is not None
    assert variables["_content"] == {
        "name": "alpha",
        "updated": 1,
    }
    record = variables["_record"]
    assert isinstance(record, RecordID)
    assert record.table_name == "_ExampleModel"
    assert record.id == "example"


@pytest.mark.asyncio
async def test_surreal_fetch_prepares_session_when_none_is_active() -> None:
    events: list[str] = []

    async def fake_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        events.append("execute")
        assert "SELECT * FROM _ExampleModel" in statement
        assert variables == {
            "_limit": 2,
            "_start": 0,
        }
        return (_row(name="alpha", updated=1),)

    database = SurrealSession(
        statement_executing=fake_execute,
        transaction_preparing=lambda: pytest.fail("unexpected transaction"),
    )

    class SessionContext:
        async def __aenter__(self) -> SurrealSession:
            events.append("enter")
            return database

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            _ = exc_type, exc_val, exc_tb
            events.append("exit")

    async with ctx.scope(
        "test.surreal.prepared_session",
        Surreal(
            session_preparing=lambda namespace, database, user, password, access: SessionContext(),
        ),
    ):
        page = await Surreal.fetch(
            _ExampleModel,
            pagination=Pagination.of(limit=1),
        )

    assert tuple(item.name for item in page.items) == ("alpha",)
    assert events == ["enter", "execute", "exit"]


@pytest.mark.asyncio
async def test_surreal_transaction_scopes_transaction_session() -> None:
    events: list[str] = []

    async def base_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        _ = statement, variables
        pytest.fail("expected transaction session")

    async def transaction_execute(
        statement: str,
        /,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Sequence[SurrealObject]:
        events.append(f"transaction:{statement}")
        assert variables == {}
        return ()

    transaction_session = SurrealSession(
        statement_executing=transaction_execute,
        transaction_preparing=lambda: pytest.fail("unexpected nested transaction"),
    )

    class TransactionContext:
        async def __aenter__(self) -> SurrealSession:
            events.append("transaction-enter")
            return transaction_session

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            _ = exc_type, exc_val, exc_tb
            events.append("transaction-exit")

    base_session = SurrealSession(
        statement_executing=base_execute,
        transaction_preparing=TransactionContext,
    )

    class SessionContext:
        async def __aenter__(self) -> SurrealSession:
            events.append("session-enter")
            return base_session

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            _ = exc_type, exc_val, exc_tb
            events.append("session-exit")

    async with ctx.scope(
        "test.surreal.transaction",
        Surreal(
            session_preparing=lambda namespace, database, user, password, access: SessionContext(),
        ),
    ):
        async with Surreal.transaction():
            await Surreal.execute("SELECT 1;")

    assert events == [
        "session-enter",
        "transaction-enter",
        "transaction:SELECT 1;",
        "transaction-exit",
        "session-exit",
    ]
