from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from haiway import Meta, Pagination

import draive.postgres.templates as postgres_templates
from draive.multimodal.templates import TemplateDeclaration
from draive.postgres.templates import PostgresTemplatesRepository


@dataclass(frozen=True)
class _FakeRow:
    identifier: str
    description: str | None
    variables: str
    meta: str

    def __getitem__(
        self,
        key: str,
    ) -> str | None:
        return getattr(self, key)


@pytest.mark.asyncio
async def test_postgres_templates_repository_templates_support_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: Sequence[_FakeRow] = (
        _FakeRow(
            identifier="welcome",
            description="Welcome template",
            variables='{"name":"Recipient name"}',
            meta='{"channel":"email"}',
        ),
        _FakeRow(
            identifier="summary",
            description=None,
            variables='{"title":"Summary title"}',
            meta="{}",
        ),
        _FakeRow(
            identifier="closing",
            description="Closing template",
            variables="{}",
            meta='{"scope":"internal"}',
        ),
    )
    page_fetches: list[tuple[object, ...]] = []

    async def fake_fetch(
        statement: str,
        /,
        *args: object,
    ) -> Sequence[_FakeRow]:
        _ = statement
        page_fetches.append(args)

        match args:
            case (3,):
                return rows

            case ("summary", 3):
                return rows[2:]

            case _:
                raise AssertionError(f"Unexpected fetch arguments: {args!r}")

    monkeypatch.setattr(postgres_templates.Postgres, "fetch", fake_fetch)

    repository = PostgresTemplatesRepository.prepare()

    page_1 = await repository.templates(Pagination.of(limit=2))
    page_2 = await repository.templates(page_1.pagination)

    assert page_1.items == (
        TemplateDeclaration(
            identifier="welcome",
            description="Welcome template",
            variables={"name": "Recipient name"},
            meta={"channel": "email"},
        ),
        TemplateDeclaration(
            identifier="summary",
            description=None,
            variables={"title": "Summary title"},
            meta={},
        ),
    )
    assert page_1.pagination.token == "summary"
    assert page_2.items == (
        TemplateDeclaration(
            identifier="closing",
            description="Closing template",
            variables={},
            meta={"scope": "internal"},
        ),
    )
    assert page_2.pagination.token is None
    assert page_fetches == [(3,), ("summary", 3)]


@pytest.mark.asyncio
async def test_postgres_templates_repository_accepts_legacy_offset_pagination_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: Sequence[_FakeRow] = (
        _FakeRow(
            identifier="closing",
            description="Closing template",
            variables="{}",
            meta='{"scope":"internal"}',
        ),
    )

    async def fake_fetch(
        statement: str,
        /,
        *args: object,
    ) -> Sequence[_FakeRow]:
        _ = statement

        assert args == ("2", 3)
        return rows

    monkeypatch.setattr(postgres_templates.Postgres, "fetch", fake_fetch)

    repository = PostgresTemplatesRepository.prepare()

    page = await repository.templates(Pagination.of(limit=2).with_token("2"))

    assert page.items == (
        TemplateDeclaration(
            identifier="closing",
            description="Closing template",
            variables={},
            meta={"scope": "internal"},
        ),
    )
    assert page.pagination.token is None


@pytest.mark.asyncio
async def test_postgres_templates_repository_define_preserves_empty_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute_calls: list[tuple[object, ...]] = []
    cache_cleared: list[bool] = []

    async def fake_execute(
        statement: str,
        /,
        *args: object,
    ) -> None:
        _ = statement
        execute_calls.append(args)

    async def fake_clear_cache() -> None:
        cache_cleared.append(True)

    monkeypatch.setattr(postgres_templates.Postgres, "execute", fake_execute)

    repository = PostgresTemplatesRepository.prepare()
    cached_load = repository._defining.__closure__[0].cell_contents
    monkeypatch.setattr(cached_load, "clear_cache", fake_clear_cache)

    await repository.define(
        TemplateDeclaration(
            identifier="welcome",
            description="",
            variables={},
            meta=Meta.empty,
        ),
        content="Hello",
    )

    assert execute_calls == [
        (
            "welcome",
            "",
            "Hello",
            "{}",
            "{}",
        )
    ]
    assert cache_cleared == [True]
