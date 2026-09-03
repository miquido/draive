from asyncio import sleep
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from haiway import Meta, Pagination, ctx

import draive.surreal.templates as surreal_templates
from draive.multimodal.templates import Template, TemplateDeclaration, TemplatesRepository
from draive.surreal import SurrealClient
from draive.surreal.templates import SurrealTemplatesRepository
from draive.surreal.types import SurrealObject


@pytest.mark.asyncio
async def test_surreal_templates_repository_templates_support_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier_rows: Sequence[SurrealObject] = cast(
        Sequence[SurrealObject],
        (
            {"value": "closing"},
            {"value": "summary"},
            {"value": "welcome"},
        ),
    )
    # revisions come back ordered by identifier and recency, only the latest
    # revision of each identifier is expected within the listing
    revision_rows: Sequence[SurrealObject] = cast(
        Sequence[SurrealObject],
        (
            {
                "identifier": "closing",
                "description": "Closing template",
                "variables": {},
                "meta": {"scope": "internal"},
            },
            {
                "identifier": "summary",
                "description": None,
                "variables": {"title": "Summary title"},
                "meta": {},
            },
            {
                "identifier": "summary",
                "description": None,
                "variables": {"title": "Old summary title"},
                "meta": {"revision": "old"},
            },
            {
                "identifier": "welcome",
                "description": "Welcome template",
                "variables": {"name": "Recipient name"},
                "meta": {"channel": "email"},
            },
        ),
    )

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        _ = statement
        if "identifiers" in variables:
            assert variables == {"identifiers": ["closing", "summary", "welcome"]}
            return revision_rows

        assert variables == {"after_identifier": None, "limit": 3}
        return identifier_rows

    monkeypatch.setattr(surreal_templates.Surreal, "execute", fake_execute)

    repository = SurrealTemplatesRepository.prepare()

    page_1 = await repository.templates(Pagination.of(limit=2))
    assert page_1.items == (
        TemplateDeclaration(
            identifier="closing",
            description="Closing template",
            variables={},
            meta=Meta.of({"scope": "internal"}),
        ),
        TemplateDeclaration(
            identifier="summary",
            description=None,
            variables={"title": "Summary title"},
            meta=Meta.empty,
        ),
    )
    assert page_1.pagination.token == "summary"


@pytest.mark.asyncio
async def test_surreal_templates_repository_uses_string_tokens_as_identifier_cursors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        assert statement
        assert variables == {"after_identifier": "1", "limit": 3}
        return ()

    monkeypatch.setattr(surreal_templates.Surreal, "execute", fake_execute)

    repository = SurrealTemplatesRepository.prepare()

    page = await repository.templates(Pagination.of(limit=2).with_token("1"))

    assert page.items == ()
    assert page.pagination.token is None


@pytest.mark.asyncio
async def test_surreal_templates_repository_listing_defaults_missing_json_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: Sequence[SurrealObject] = cast(
        Sequence[SurrealObject],
        (
            {
                "identifier": "welcome",
                "description": "Welcome template",
                "variables": None,
                "meta": None,
            },
        ),
    )

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        _ = statement
        if "identifiers" in variables:
            assert variables == {"identifiers": ["welcome"]}
            return rows

        assert variables == {"after_identifier": None, "limit": 2}
        return cast(Sequence[SurrealObject], ({"value": "welcome"},))

    monkeypatch.setattr(surreal_templates.Surreal, "execute", fake_execute)

    repository = SurrealTemplatesRepository.prepare()

    page = await repository.templates(Pagination.of(limit=1))

    assert page.items == (
        TemplateDeclaration(
            identifier="welcome",
            description="Welcome template",
            variables={},
            meta=Meta.empty,
        ),
    )
    assert page.pagination.token is None


@pytest.mark.asyncio
async def test_surreal_templates_repository_loads_latest_template_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: Sequence[SurrealObject] = cast(
        Sequence[SurrealObject],
        (
            {"content": "New content"},
            {"content": "Old content"},
        ),
    )
    execute_calls: list[Mapping[str, Any]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        assert "ORDER BY" in statement
        assert variables == {"identifier": "welcome"}
        execute_calls.append(cast(Mapping[str, Any], variables))
        return rows

    monkeypatch.setattr(surreal_templates.Surreal, "execute", fake_execute)

    repository = SurrealTemplatesRepository.prepare()

    assert await repository.load(Template.of("welcome")) == "New content"
    assert execute_calls == [{"identifier": "welcome"}]


@pytest.mark.asyncio
async def test_surreal_templates_repository_define_creates_new_history_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute_calls: list[Mapping[str, Any]] = []

    current_content: str = "cached content"

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        nonlocal current_content
        if "SELECT" in statement:
            execute_calls.append(cast(Mapping[str, Any], variables))
            return (cast(SurrealObject, {"content": current_content}),)

        assert "CREATE templates CONTENT" in statement
        execute_calls.append(cast(Mapping[str, Any], variables))
        current_content = cast(str, variables["content"])
        return ()

    monkeypatch.setattr(surreal_templates.Surreal, "execute", fake_execute)

    repository = SurrealTemplatesRepository.prepare(cache_limit=1, cache_expiration=3600.0)

    first_load = await repository.load(Template.of("welcome"))
    second_load = await repository.load(Template.of("welcome"))
    assert first_load == "cached content"
    assert second_load == "cached content"

    await repository.define(
        TemplateDeclaration(
            identifier="welcome",
            description="Welcome template",
            variables={"name": "Recipient name"},
            meta=Meta.of({"channel": "email"}),
        ),
        content="Hello {%name%}",
    )

    variables = execute_calls[1]
    assert variables["identifier"] == "welcome"
    assert isinstance(variables["updated"], datetime)
    assert cast(datetime, variables["updated"]).tzinfo == UTC
    assert variables["description"] == "Welcome template"
    assert variables["content"] == "Hello {%name%}"
    assert variables["variables"] == {"name": "Recipient name"}
    assert variables["meta"] == {"channel": "email"}
    updated_load = await repository.load(Template.of("welcome"))
    assert updated_load == "Hello {%name%}"

    assert len(execute_calls) == 3
    assert "identifier" in execute_calls[0]
    assert "identifier" in execute_calls[1]
    assert "identifier" in execute_calls[2]


@pytest.mark.asyncio
async def test_surreal_embedded_templates_listing_keeps_every_identifier() -> None:
    """Live regression test: selecting the latest revision of each identifier used to
    rely on a `$parent`-correlated subquery, which a SurrealDB server evaluates
    unreliably - it silently dropped a varying subset of the templates on each run.
    """
    async with SurrealClient(
        url="mem://",
        namespace="test_surreal_templates",
        database="embedded_templates",
    ) as surreal:
        repository: TemplatesRepository = SurrealTemplatesRepository.prepare(cache_expiration=0.001)
        async with ctx.scope("test.surreal.templates", surreal, repository):
            await SurrealTemplatesRepository.migrate()
            for identifier in ("greeting", "farewell", "summary"):
                await repository.define(
                    TemplateDeclaration(
                        identifier=identifier,
                        description=f"{identifier} template",
                        variables={"name": "recipient"},
                        meta=Meta.empty,
                    ),
                    content=f"{identifier} {{%name%}}",
                )

            await sleep(0.01)  # revisions are ordered by their timestamp
            await repository.define(
                TemplateDeclaration(
                    identifier="greeting",
                    description="greeting revision",
                    variables={"name": "recipient"},
                    meta=Meta.empty,
                ),
                content="hello {%name%}",
            )

            listed = await repository.templates()
            assert [declaration.identifier for declaration in listed.items] == [
                "farewell",
                "greeting",
                "summary",
            ]
            greeting = next(
                declaration for declaration in listed.items if declaration.identifier == "greeting"
            )
            assert greeting.description == "greeting revision"
            assert await repository.load(Template.of("greeting")) == "hello {%name%}"

            first_page = await repository.templates(Pagination.of(limit=2))
            assert [declaration.identifier for declaration in first_page.items] == [
                "farewell",
                "greeting",
            ]
            assert first_page.pagination.token == "greeting"

            second_page = await repository.templates(first_page.pagination)
            assert [declaration.identifier for declaration in second_page.items] == ["summary"]
            assert second_page.pagination.token is None


@pytest.mark.asyncio
async def test_surreal_templates_repository_migration_defines_table_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        _ = variables
        statements.append(statement)
        return ()

    monkeypatch.setattr(surreal_templates.Surreal, "execute", fake_execute)

    await SurrealTemplatesRepository.migrate()

    assert statements == [
        "DEFINE TABLE IF NOT EXISTS templates SCHEMALESS TYPE NORMAL;",
        "DEFINE INDEX IF NOT EXISTS templates_identifier_idx "
        "ON TABLE templates FIELDS identifier, updated;",
    ]
