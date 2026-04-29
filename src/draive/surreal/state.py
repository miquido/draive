import sys
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Literal, cast, overload
from uuid import UUID, uuid4

from haiway import (
    AttributePath,
    AttributeRequirement,
    Paginated,
    Pagination,
    State,
    ctx,
    statemethod,
)
from surrealdb import RecordID

from draive.surreal.filters import prepare_filter
from draive.surreal.schema import SurrealTableKind, surreal_field_definitions
from draive.surreal.types import (
    SurrealID,
    SurrealObject,
    SurrealSessionContext,
    SurrealSessionPreparing,
    SurrealStatementExecuting,
    SurrealTransactionContext,
    SurrealTransactionPreparing,
    SurrealValue,
)
from draive.surreal.utils import pagination_offset

__all__ = (
    "Surreal",
    "SurrealSession",
)


class SurrealSession(State):
    @overload
    @classmethod
    async def execute(
        cls,
        statement: str,
        /,
        **variables: SurrealValue,
    ) -> Sequence[SurrealObject]: ...

    @overload
    async def execute(
        self,
        statement: str,
        /,
        **variables: SurrealValue,
    ) -> Sequence[SurrealObject]: ...

    @statemethod
    async def execute(
        self,
        statement: str,
        /,
        **variables: SurrealValue,
    ) -> Sequence[SurrealObject]:
        return await self._statement_executing(
            statement,
            variables=variables,
        )

    @overload
    @classmethod
    def transaction(  # pyright: ignore[reportInconsistentOverload]
        # it seems to be pyright limitation and false positive
        cls,
    ) -> SurrealTransactionContext: ...

    @overload
    def transaction(self) -> SurrealTransactionContext: ...

    @statemethod
    def transaction(self) -> SurrealTransactionContext:
        return self._transaction_preparing()

    @overload
    @classmethod
    async def create[Model: State](
        cls,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str | None = None,
    ) -> str: ...

    @overload
    async def create[Model: State](
        self,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str | None = None,
    ) -> str: ...

    @statemethod
    async def create[Model: State](
        self,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str | None = None,
    ) -> str:
        if identifier is None:
            identifier = uuid4().hex  # generate new

        elif isinstance(identifier, UUID):
            identifier = identifier.hex

        elif not isinstance(identifier, str):
            identifier = identifier(value)

        await self._statement_executing(
            "CREATE $_record CONTENT $_content;",
            variables={
                "_record": RecordID(value.__class__.__name__, identifier),
                "_content": value.to_mapping(),
            },
        )

        return identifier

    @overload
    @classmethod
    async def upsert[Model: State](
        cls,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str,
    ) -> str: ...

    @overload
    async def upsert[Model: State](
        self,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str,
    ) -> str: ...

    @statemethod
    async def upsert[Model: State](
        self,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str,
    ) -> str:
        if isinstance(identifier, UUID):
            identifier = identifier.hex

        elif not isinstance(identifier, str):
            identifier = identifier(value)

        await self._statement_executing(
            "UPSERT $_record CONTENT $_content;",
            variables={
                "_record": RecordID(value.__class__.__name__, identifier),
                "_content": value.to_mapping(),
            },
        )

        return identifier

    @overload
    @classmethod
    async def fetch[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        pagination: Pagination | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        order: AttributePath[Model, object] | None = None,
    ) -> Paginated[Model]: ...

    @overload
    async def fetch[Model: State](
        self,
        model: type[Model],
        /,
        *,
        pagination: Pagination | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        order: AttributePath[Model, object] | None = None,
    ) -> Paginated[Model]: ...

    @statemethod
    async def fetch[Model: State](
        self,
        model: type[Model],
        /,
        *,
        pagination: Pagination | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        order: AttributePath[Model, object] | None = None,
    ) -> Paginated[Model]:
        pagination = pagination or Pagination.of(limit=32)
        if pagination.limit <= 0:
            return Paginated[Model].of(
                (),
                pagination=pagination.with_token(None),
            )

        start_at: int = pagination_offset(pagination)
        filter_clause: str
        filter_variables: Mapping[str, SurrealValue]
        filter_clause, filter_variables = prepare_filter(requirements)
        fetch_limit: int = pagination.limit + 1

        records: Sequence[SurrealObject] = await self._statement_executing(
            f"SELECT * FROM {model.__name__}"  # nosec: B608
            f"{f' WHERE {filter_clause}' if filter_clause else ''}"
            f"{f' ORDER BY {order!s} DESC, id DESC' if order is not None else ''}"
            " LIMIT $_limit START AT $_start;",
            variables={
                **filter_variables,
                "_limit": fetch_limit,
                "_start": start_at,
            },
        )

        next_token: int | None = None
        if len(records) > pagination.limit:
            next_token = start_at + pagination.limit

        return Paginated[Model].of(
            tuple(model.from_mapping(record) for record in records[: pagination.limit]),
            pagination=pagination.with_token(next_token),
        )

    @overload
    @classmethod
    async def delete[Model: State](
        cls,
        table: type[Model] | str,
        /,
        identifier: UUID | str | None = None,
        *,
        requirements: AttributeRequirement[Model] | None = None,
    ) -> None: ...

    @overload
    async def delete[Model: State](
        self,
        table: type[Model] | str,
        /,
        identifier: UUID | str | None = None,
        *,
        requirements: AttributeRequirement[Model] | None = None,
    ) -> None: ...

    @statemethod
    async def delete[Model: State](
        self,
        table: type[Model] | str,
        /,
        identifier: UUID | str | None = None,
        *,
        requirements: AttributeRequirement[Model] | None = None,
    ) -> None:
        if identifier is not None:
            if requirements is not None:
                raise ValueError("Deleting by identifier does not support additional requirements")

            if isinstance(identifier, UUID):
                identifier = identifier.hex

            await self._statement_executing(
                "DELETE $_record;",
                variables={
                    "_record": RecordID(
                        table if isinstance(table, str) else table.__name__,
                        identifier,
                    ),
                },
            )

        elif requirements is not None:
            filter_clause, filter_variables = prepare_filter(requirements)
            await self._statement_executing(
                f"DELETE $_table WHERE {filter_clause};",
                variables={
                    "_table": table if isinstance(table, str) else table.__name__,
                    **filter_variables,
                },
            )

        else:
            await self._statement_executing(
                "DELETE $_table;",
                variables={
                    "_table": table if isinstance(table, str) else table.__name__,
                },
            )

    @overload
    @classmethod
    async def define_table[Model: State](
        cls,
        table: type[Model] | str,
        /,
        *,
        schema: bool = False,
        kind: SurrealTableKind = "NORMAL",
    ) -> None: ...

    @overload
    async def define_table[Model: State](
        self,
        table: type[Model] | str,
        /,
        *,
        schema: bool = False,
        kind: SurrealTableKind = "NORMAL",
    ) -> None: ...

    @statemethod
    async def define_table[Model: State](
        self,
        table: type[Model] | str,
        /,
        *,
        schema: bool = False,
        kind: SurrealTableKind = "NORMAL",
    ) -> None:
        await self._statement_executing(
            f"DEFINE TABLE IF NOT EXISTS "
            f"{table if isinstance(table, str) else table.__name__} "
            f"{'SCHEMAFULL' if schema else 'SCHEMALESS'} TYPE {kind};",
            variables={},
        )

        if schema and not isinstance(table, str):
            for statement in surreal_field_definitions(
                table,
                table=table.__name__,
            ):
                await self._statement_executing(
                    statement,
                    variables={},
                )

    @overload
    @classmethod
    async def define_relation[Edge: State, Source: State, Target: State](
        cls,
        relation: type[Edge] | str,
        /,
        *,
        from_table: type[Source] | str,
        to_table: type[Target] | str,
        schema: bool = False,
    ) -> None: ...

    @overload
    async def define_relation[Edge: State, Source: State, Target: State](
        self,
        relation: type[Edge] | str,
        /,
        *,
        from_table: type[Source] | str,
        to_table: type[Target] | str,
        schema: bool = False,
    ) -> None: ...

    @statemethod
    async def define_relation[Edge: State, Source: State, Target: State](
        self,
        relation: type[Edge] | str,
        /,
        *,
        from_table: type[Source] | str,
        to_table: type[Target] | str,
        schema: bool = False,
    ) -> None:
        relation_name: str = relation if isinstance(relation, str) else relation.__name__
        from_name: str = from_table if isinstance(from_table, str) else from_table.__name__
        to_name: str = to_table if isinstance(to_table, str) else to_table.__name__
        await self._statement_executing(
            f"DEFINE TABLE IF NOT EXISTS {relation_name} "
            f"{'SCHEMAFULL' if schema else 'SCHEMALESS'} "
            f"TYPE RELATION FROM {from_name} TO {to_name};",
            variables={},
        )

        if schema and not isinstance(relation, str):
            for statement in surreal_field_definitions(relation, table=relation_name):
                await self._statement_executing(
                    statement,
                    variables={},
                )

    @overload
    @classmethod
    async def relate[Edge: State, Source: State, Target: State](
        cls,
        source_table: type[Source] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target_table: type[Target] | str,
        target_identifier: UUID | str,
        /,
        *,
        content: Edge | Mapping[str, SurrealValue] | None = None,
    ) -> str | None: ...

    @overload
    async def relate[Edge: State, Source: State, Target: State](
        self,
        source_table: type[Source] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target_table: type[Target] | str,
        target_identifier: UUID | str,
        /,
        *,
        content: Edge | Mapping[str, SurrealValue] | None = None,
    ) -> str | None: ...

    @statemethod
    async def relate[Edge: State, Source: State, Target: State](
        self,
        source_table: type[Source] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target_table: type[Target] | str,
        target_identifier: UUID | str,
        /,
        *,
        content: Edge | Mapping[str, SurrealValue] | None = None,
    ) -> str | None:
        relation_name: str = relation if isinstance(relation, str) else relation.__name__
        rows: Sequence[SurrealObject]
        variables: dict[str, SurrealValue] = {
            "_source": RecordID(
                source_table if isinstance(source_table, str) else source_table.__name__,
                _surreal_record(source_identifier),
            ),
            "_target": RecordID(
                target_table if isinstance(target_table, str) else target_table.__name__,
                _surreal_record(target_identifier),
            ),
        }

        if content is None:
            rows = await self._statement_executing(
                f"RELATE $_source->{relation_name}->$_target RETURN id;",  # nosec: B608
                variables=variables,
            )

        else:
            variables["_content"] = content.to_mapping() if isinstance(content, State) else content
            rows = await self._statement_executing(
                f"RELATE $_source->{relation_name}->$_target CONTENT $_content RETURN id;",  # nosec: B608
                variables=variables,
            )

        if not rows:
            return None

        identifier: SurrealValue | None = rows[0].get("id")
        if isinstance(identifier, SurrealID):
            return str(identifier.record)

        if isinstance(identifier, RecordID):
            return str(identifier.id)

        if isinstance(identifier, str):
            return identifier

        return None

    @overload
    @classmethod
    async def related[Model: State, Edge: State](
        cls,
        source_table: type[State] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target: type[Model],
        /,
        *,
        direction: Literal["out", "in"] = "out",
        pagination: Pagination | None = None,
    ) -> Paginated[Model]: ...

    @overload
    async def related[Model: State, Edge: State](
        self,
        source_table: type[State] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target: type[Model],
        /,
        *,
        direction: Literal["out", "in"] = "out",
        pagination: Pagination | None = None,
    ) -> Paginated[Model]: ...

    @statemethod
    async def related[Model: State, Edge: State](
        self,
        source_table: type[State] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target: type[Model],
        /,
        *,
        direction: Literal["out", "in"] = "out",
        pagination: Pagination | None = None,
    ) -> Paginated[Model]:
        pagination = pagination or Pagination.of(limit=32)
        if pagination.limit <= 0:
            return Paginated[Model].of(
                (),
                pagination=pagination.with_token(None),
            )

        relation_name: str = relation if isinstance(relation, str) else relation.__name__
        source_field: str
        target_field: str
        match direction:
            case "out":
                source_field = "in"
                target_field = "out"

            case "in":
                source_field = "out"
                target_field = "in"

        start_at: int = pagination_offset(pagination)
        fetch_limit: int = pagination.limit + 1
        rows: Sequence[SurrealObject] = await self._statement_executing(
            f"SELECT {target_field}.* AS value FROM {relation_name} "  # nosec: B608
            f"WHERE {source_field} = $_source "
            "ORDER BY id DESC LIMIT $_limit START AT $_start;",
            variables={
                "_source": RecordID(
                    source_table if isinstance(source_table, str) else source_table.__name__,
                    _surreal_record(source_identifier),
                ),
                "_limit": fetch_limit,
                "_start": start_at,
            },
        )

        page_records: Sequence[SurrealObject] = rows[: pagination.limit]
        next_token: int | None = None
        if len(rows) > pagination.limit:
            next_token = start_at + pagination.limit

        return Paginated[Model].of(
            tuple(
                target.from_mapping(cast(Mapping[str, object], record["value"]))
                for record in page_records
            ),
            pagination=pagination.with_token(next_token),
        )

    _statement_executing: SurrealStatementExecuting
    _transaction_preparing: SurrealTransactionPreparing

    def __init__(
        self,
        statement_executing: SurrealStatementExecuting,
        transaction_preparing: SurrealTransactionPreparing,
    ) -> None:
        super().__init__(
            _statement_executing=statement_executing,
            _transaction_preparing=transaction_preparing,
        )


class Surreal(State):
    @overload
    @classmethod
    def prepare_session(  # pyright: ignore[reportInconsistentOverload]
        # it seems to be pyright limitation and false positive
        cls,
        namespace: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        access: str | None = None,
    ) -> SurrealSessionContext: ...

    @overload
    def prepare_session(
        self,
        namespace: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        access: str | None = None,
    ) -> SurrealSessionContext: ...

    @statemethod
    def prepare_session(
        self,
        namespace: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        access: str | None = None,
    ) -> SurrealSessionContext:
        if ctx.contains_state(SurrealSession):
            raise RuntimeError("Recursive Surreal session acquiring is forbidden")

        return self._session_preparing(
            namespace=namespace,
            database=database,
            user=user,
            password=password,
            access=access,
        )

    @classmethod
    def transaction(cls) -> SurrealTransactionContext:
        if ctx.contains_state(SurrealSession):
            return SurrealSession.transaction()

        return _SurrealTransactionContext(session=cls.prepare_session())

    @classmethod
    async def execute(
        cls,
        statement: str,
        /,
        **variables: SurrealValue,
    ) -> Sequence[SurrealObject]:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.execute(
                statement,
                **variables,
            )

        async with cls.prepare_session() as session:
            return await session.execute(
                statement,
                **variables,
            )

    @classmethod
    async def create[Model: State](
        cls,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str | None = None,
    ) -> str:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.create(
                value,
                identifier=identifier,
            )

        async with cls.prepare_session() as session:
            return await session.create(
                value,
                identifier=identifier,
            )

    @classmethod
    async def upsert[Model: State](
        cls,
        value: Model,
        /,
        *,
        identifier: AttributePath[Model, str] | UUID | str,
    ) -> str:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.upsert(
                value,
                identifier=identifier,
            )

        async with cls.prepare_session() as session:
            return await session.upsert(
                value,
                identifier=identifier,
            )

    @classmethod
    async def fetch[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        pagination: Pagination | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        order: AttributePath[Model, object] | None = None,
    ) -> Paginated[Model]:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.fetch(
                model,
                pagination=pagination,
                requirements=requirements,
                order=order,
            )

        async with cls.prepare_session() as session:
            return await session.fetch(
                model,
                pagination=pagination,
                requirements=requirements,
                order=order,
            )

    @classmethod
    async def delete[Model: State](
        cls,
        table: type[Model] | str,
        /,
        identifier: UUID | str | None = None,
        *,
        requirements: AttributeRequirement[Model] | None = None,
    ) -> None:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.delete(
                table,
                identifier,
                requirements=requirements,
            )

        async with cls.prepare_session() as session:
            return await session.delete(
                table,
                identifier,
                requirements=requirements,
            )

    @classmethod
    async def define_table[Model: State](
        cls,
        table: type[Model] | str,
        /,
        *,
        schema: bool = False,
        kind: SurrealTableKind = "NORMAL",
    ) -> None:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.define_table(
                table,
                schema=schema,
                kind=kind,
            )

        async with cls.prepare_session() as session:
            return await session.define_table(
                table,
                schema=schema,
                kind=kind,
            )

    @classmethod
    async def define_relation[Edge: State, Source: State, Target: State](
        cls,
        relation: type[Edge] | str,
        /,
        *,
        from_table: type[Source] | str,
        to_table: type[Target] | str,
        schema: bool = False,
    ) -> None:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.define_relation(
                relation,
                from_table=from_table,
                to_table=to_table,
                schema=schema,
            )

        async with cls.prepare_session() as session:
            return await session.define_relation(
                relation,
                from_table=from_table,
                to_table=to_table,
                schema=schema,
            )

    @classmethod
    async def relate[Edge: State, Source: State, Target: State](
        cls,
        source_table: type[Source] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target_table: type[Target] | str,
        target_identifier: UUID | str,
        /,
        *,
        content: Edge | Mapping[str, SurrealValue] | None = None,
    ) -> str | None:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.relate(
                source_table,
                source_identifier,
                relation,
                target_table,
                target_identifier,
                content=content,
            )

        async with cls.prepare_session() as session:
            return await session.relate(
                source_table,
                source_identifier,
                relation,
                target_table,
                target_identifier,
                content=content,
            )

    @classmethod
    async def related[Model: State, Edge: State](
        cls,
        source_table: type[State] | str,
        source_identifier: UUID | str,
        relation: type[Edge] | str,
        target: type[Model],
        /,
        *,
        direction: Literal["out", "in"] = "out",
        pagination: Pagination | None = None,
    ) -> Paginated[Model]:
        if ctx.contains_state(SurrealSession):
            return await SurrealSession.related(
                source_table,
                source_identifier,
                relation,
                target,
                direction=direction,
                pagination=pagination,
            )

        async with cls.prepare_session() as session:
            return await session.related(
                source_table,
                source_identifier,
                relation,
                target,
                direction=direction,
                pagination=pagination,
            )

    _session_preparing: SurrealSessionPreparing

    def __init__(
        self,
        session_preparing: SurrealSessionPreparing,
    ) -> None:
        super().__init__(_session_preparing=session_preparing)


class _SurrealTransactionContext:
    __slots__ = (
        "_scope_context",
        "_session",
        "_session_context",
        "_transaction_context",
    )

    def __init__(
        self,
        *,
        session: SurrealSessionContext,
    ) -> None:
        self._session_context: SurrealSessionContext = session
        self._session: SurrealSession | None = None
        self._scope_context: AbstractAsyncContextManager[str] | None = None
        self._transaction_context: SurrealTransactionContext | None = None

    async def __aenter__(self) -> SurrealSession:
        session: SurrealSession = await self._session_context.__aenter__()
        self._session = session

        try:
            transaction_context: SurrealTransactionContext = session.transaction()
            self._transaction_context = transaction_context
            transaction_session: SurrealSession = await transaction_context.__aenter__()

            try:
                scope_context: AbstractAsyncContextManager[str] = ctx.scope(
                    "surreal.transaction",
                    transaction_session,
                )
                self._scope_context = scope_context
                await scope_context.__aenter__()
                return transaction_session

            except BaseException:
                self._scope_context = None
                await transaction_context.__aexit__(*sys.exc_info())
                raise

        except BaseException:
            self._transaction_context = None
            self._session = None
            await self._session_context.__aexit__(*sys.exc_info())
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        scope_context: AbstractAsyncContextManager[str] | None = self._scope_context
        transaction_context: SurrealTransactionContext | None = self._transaction_context
        self._scope_context = None
        self._transaction_context = None
        self._session = None
        try:
            if scope_context is not None:
                await scope_context.__aexit__(
                    exc_type,
                    exc_val,
                    exc_tb,
                )

        finally:
            try:
                if transaction_context is not None:
                    await transaction_context.__aexit__(
                        exc_type,
                        exc_val,
                        exc_tb,
                    )

            finally:
                await self._session_context.__aexit__(
                    exc_type,
                    exc_val,
                    exc_tb,
                )


def _surreal_record(
    identifier: UUID | str,
    /,
) -> str:
    if isinstance(identifier, UUID):
        return identifier.hex

    return identifier
