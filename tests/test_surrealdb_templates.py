from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from haiway import Meta, Pagination

import draive.surreal.templates as surreal_templates
from draive.multimodal.templates import Template, TemplateDeclaration
from draive.surreal.templates import SurrealTemplatesRepository
from draive.surreal.types import SurrealObject


@pytest.mark.asyncio
async def test_surrealdb_templates_repository_templates_support_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_rows: Sequence[SurrealObject] = cast(
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
        assert variables == {"after_identifier": None, "limit": 3}
        return initial_rows

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
async def test_surrealdb_templates_repository_uses_string_tokens_as_identifier_cursors(
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
async def test_surrealdb_templates_repository_listing_defaults_missing_json_fields(
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
        assert variables == {"after_identifier": None, "limit": 2}
        return rows

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
async def test_surrealdb_templates_repository_loads_latest_template_content(
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
async def test_surrealdb_templates_repository_define_creates_new_history_entry(
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
