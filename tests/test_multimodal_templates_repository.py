from pathlib import Path

import pytest
from haiway import Meta, Pagination, ctx

from draive.multimodal.content import MultimodalContent
from draive.multimodal.templates import (
    Template,
    TemplateDeclaration,
    TemplateInvalid,
    TemplateMissing,
    TemplatesRepository,
)
from draive.multimodal.text import TextContent


def test_template_missing_exposes_identifier() -> None:
    exception = TemplateMissing(identifier="welcome")

    assert exception.identifier == "welcome"
    assert str(exception) == "Missing template - welcome"


def test_template_of_uses_empty_defaults() -> None:
    template = Template.of("welcome")

    assert template.identifier == "welcome"
    assert template.arguments == {}
    assert template.meta == Meta.empty


def test_template_with_arguments_returns_self_when_no_values() -> None:
    template = Template.of("welcome")

    assert template.with_arguments() is template


def test_template_with_arguments_merges_and_overrides_values() -> None:
    template = Template.of(
        "welcome",
        arguments={
            "name": "Ada",
            "closing": "Regards",
        },
    )

    updated = template.with_arguments(
        closing="Bye",
        extra="!",
    )

    assert updated is not template
    assert updated.arguments == {
        "name": "Ada",
        "closing": "Bye",
        "extra": "!",
    }
    assert template.arguments == {
        "name": "Ada",
        "closing": "Regards",
    }


def test_template_with_meta_returns_self_for_empty_meta() -> None:
    template = Template.of("welcome")

    assert template.with_meta({}) is template


def test_template_with_meta_merges_meta_values() -> None:
    template = Template.of("welcome", meta={"scope": "base", "priority": "low"})

    updated = template.with_meta({"priority": "high", "channel": "email"})

    assert updated is not template
    assert updated.meta == Meta(
        {
            "scope": "base",
            "priority": "high",
            "channel": "email",
        }
    )


def test_template_declaration_of_uses_empty_defaults() -> None:
    declaration = TemplateDeclaration.of("welcome")

    assert declaration.identifier == "welcome"
    assert declaration.description is None
    assert declaration.variables == {}
    assert declaration.meta == Meta.empty


@pytest.mark.asyncio
async def test_volatile_repository_lists_parsed_declarations() -> None:
    repository = TemplatesRepository.volatile(
        welcome="Hello {%name%}",
        summary="{%title%}: {%body%}",
    )

    declarations = await repository.templates()

    assert declarations.items == (
        TemplateDeclaration(
            identifier="welcome",
            description=None,
            variables={"name": "name"},
            meta=Meta.empty,
        ),
        TemplateDeclaration(
            identifier="summary",
            description=None,
            variables={"title": "title", "body": "body"},
            meta=Meta.empty,
        ),
    )
    assert declarations.pagination == Pagination.of(limit=32)
    assert repository.meta == Meta({"source": "volatile"})


@pytest.mark.asyncio
async def test_resolve_returns_raw_content_without_arguments() -> None:
    repository = TemplatesRepository()

    result = await repository.resolve("Hello world")

    assert result == MultimodalContent.of("Hello world")


@pytest.mark.asyncio
async def test_resolve_raises_for_nested_template_passed_in_runtime_arguments() -> None:
    repository = TemplatesRepository.volatile(
        nested="dear reader",
    )

    with pytest.raises(RecursionError):
        await repository.resolve(
            "Hello {%name%}!",
            arguments={"name": Template.of("nested")},
        )


@pytest.mark.asyncio
async def test_resolve_raises_when_nested_template_depends_on_outer_arguments() -> None:
    repository = TemplatesRepository.volatile(
        name="Dr. {%person%}",
    )

    with pytest.raises(RecursionError):
        await repository.resolve(
            "Hello {%name%}",
            arguments={
                "name": Template.of("name"),
                "person": "Ada",
            },
        )


@pytest.mark.asyncio
async def test_resolve_raw_template_raises_for_missing_argument_with_empty_mapping() -> None:
    repository = TemplatesRepository()

    with pytest.raises(KeyError, match="Missing template argument: name"):
        await repository.resolve("Hello {%name%}", arguments={})


@pytest.mark.asyncio
async def test_resolve_raw_template_eagerly_resolves_unused_template_argument() -> None:
    repository = TemplatesRepository()

    with pytest.raises(TemplateMissing, match="Missing template - missing"):
        await repository.resolve(
            "Hello world",
            arguments={"unused": Template.of("missing")},
        )


@pytest.mark.asyncio
async def test_resolve_uses_loaded_template_defaults_and_call_overrides() -> None:
    repository = TemplatesRepository.volatile(
        welcome="Hello {%name%}{%suffix%}",
    )

    result = await repository.resolve(
        Template.of(
            "welcome",
            arguments={"name": "Ada", "suffix": "."},
        ),
        arguments={"suffix": "!"},
    )

    assert result.to_str() == "Hello Ada!"


@pytest.mark.asyncio
async def test_resolve_raises_for_nested_template_defaults() -> None:
    repository = TemplatesRepository.volatile(
        welcome="Hello {%name%}",
        name="Dr. {%person%}",
    )

    with pytest.raises(RecursionError):
        await repository.resolve(
            Template.of(
                "welcome",
                arguments={
                    "name": Template.of(
                        "name",
                        arguments={"person": "Ada"},
                    )
                },
            )
        )


@pytest.mark.asyncio
async def test_resolve_uses_default_when_template_is_missing() -> None:
    repository = TemplatesRepository()

    result = await repository.resolve(
        Template.of("missing", arguments={"name": "Ada"}),
        default="Fallback for {%name%}",
    )

    assert result == MultimodalContent.of("Fallback for Ada")


@pytest.mark.asyncio
async def test_resolve_raises_when_template_is_missing_and_no_default() -> None:
    repository = TemplatesRepository()

    with pytest.raises(TemplateMissing, match="Missing template - missing") as error:
        await repository.resolve(Template.of("missing"))

    assert error.value.identifier == "missing"


@pytest.mark.asyncio
async def test_resolve_str_returns_raw_string_without_arguments() -> None:
    repository = TemplatesRepository()

    result = await repository.resolve_str("Hello world")

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_resolve_str_raises_for_nested_template_passed_in_runtime_arguments() -> None:
    repository = TemplatesRepository.volatile(
        nested="Ada",
        welcome="Hello {%name%}",
    )

    with pytest.raises(RecursionError):
        await repository.resolve_str(
            Template.of("welcome"),
            arguments={"name": Template.of("nested")},
        )


@pytest.mark.asyncio
async def test_resolve_str_raises_when_nested_template_depends_on_outer_arguments() -> None:
    repository = TemplatesRepository.volatile(
        name="Dr. {%person%}",
        welcome="Hello {%name%}",
    )

    with pytest.raises(RecursionError):
        await repository.resolve_str(
            Template.of("welcome"),
            arguments={
                "name": Template.of("name"),
                "person": "Ada",
            },
        )


@pytest.mark.asyncio
async def test_resolve_str_raw_template_raises_for_missing_argument_with_empty_mapping() -> None:
    repository = TemplatesRepository()

    with pytest.raises(KeyError, match="Missing template argument: name"):
        await repository.resolve_str("Hello {%name%}", arguments={})


@pytest.mark.asyncio
async def test_resolve_str_raw_template_eagerly_resolves_unused_template_argument() -> None:
    repository = TemplatesRepository()

    with pytest.raises(TemplateMissing, match="Missing template - missing"):
        await repository.resolve_str(
            "Hello world",
            arguments={"unused": Template.of("missing")},
        )


@pytest.mark.asyncio
async def test_resolve_str_uses_default_when_template_is_missing() -> None:
    repository = TemplatesRepository()

    result = await repository.resolve_str(
        Template.of("missing", arguments={"name": "Ada"}),
        default="Hello {%name%}",
    )

    assert result == "Hello Ada"


@pytest.mark.asyncio
async def test_resolve_str_raises_when_template_is_missing_and_no_default() -> None:
    repository = TemplatesRepository()

    with pytest.raises(TemplateMissing, match="Missing template - missing") as error:
        await repository.resolve_str(Template.of("missing"))

    assert error.value.identifier == "missing"


@pytest.mark.asyncio
async def test_load_returns_template_content() -> None:
    repository = TemplatesRepository.volatile(
        welcome="Hello {%name%}",
    )

    loaded = await repository.load(Template.of("welcome"))

    assert loaded == "Hello {%name%}"


@pytest.mark.asyncio
async def test_load_raises_for_missing_template() -> None:
    repository = TemplatesRepository()

    with pytest.raises(TemplateMissing, match="Missing template - welcome") as error:
        await repository.load(Template.of("welcome"))

    assert error.value.identifier == "welcome"


@pytest.mark.asyncio
async def test_define_persists_template_in_volatile_repository() -> None:
    repository = TemplatesRepository.volatile()
    declaration = TemplateDeclaration.of(
        "welcome",
        description="Greeting template",
        variables={"name": "Recipient name"},
        meta={"scope": "emails"},
    )

    await repository.define(declaration, content="Hello {%name%}")

    assert await repository.load(Template.of("welcome")) == "Hello {%name%}"
    assert (await repository.templates()).items == (declaration,)


@pytest.mark.asyncio
async def test_define_raises_for_mismatched_declared_variables() -> None:
    repository = TemplatesRepository.volatile()
    declaration = TemplateDeclaration.of(
        "welcome",
        variables={"wrong": "Wrong variable"},
    )

    with pytest.raises(TemplateInvalid, match="Invalid template") as error:
        await repository.define(declaration, content="Hello {%name%}")

    assert error.value.identifier == "welcome"
    assert "declared variables do not match template content" in error.value.description


@pytest.mark.asyncio
async def test_file_repository_persists_and_reloads_templates(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    repository = TemplatesRepository.file(path)
    declaration = TemplateDeclaration.of(
        "welcome",
        description="Greeting template",
        variables={"name": "Recipient name"},
        meta={"scope": "emails"},
    )

    async with ctx.scope("test"):
        await repository.define(declaration, content="Hello {%name%}")

    reloaded = TemplatesRepository.file(path)

    async with ctx.scope("test"):
        assert (await reloaded.templates()).items == (declaration,)
        assert await reloaded.load(Template.of("welcome")) == "Hello {%name%}"
    assert reloaded.meta == Meta({"source": str(path)})


@pytest.mark.asyncio
async def test_file_repository_ignores_invalid_storage_elements(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    path.write_text(
        (
            '[{"identifier":"valid","description":null,"variables":{"name":"name"},'
            '"content":"Hello {%name%}","meta":{"scope":"ok"}},'
            '{"identifier":"invalid"}]'
        ),
        encoding="utf-8",
    )
    repository = TemplatesRepository.file(path)

    async with ctx.scope("test"):
        declarations = await repository.templates()

    assert declarations.items == (
        TemplateDeclaration(
            identifier="valid",
            description=None,
            variables={"name": "name"},
            meta=Meta({"scope": "ok"}),
        ),
    )
    async with ctx.scope("test"):
        assert await repository.load(Template.of("valid")) == "Hello {%name%}"


@pytest.mark.asyncio
async def test_file_repository_invalid_element_does_not_hide_following_valid_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "templates.json"
    path.write_text(
        (
            '[{"identifier":"invalid","description":null,"variables":{"name":123},'
            '"content":"Broken","meta":{}},'
            '{"identifier":"valid","description":null,"variables":{"name":"name"},'
            '"content":"Hello {%name%}","meta":{"scope":"ok"}}]'
        ),
        encoding="utf-8",
    )
    repository = TemplatesRepository.file(path)

    async with ctx.scope("test"):
        declarations = await repository.templates()

    assert declarations.items == (
        TemplateDeclaration(
            identifier="valid",
            description=None,
            variables={"name": "name"},
            meta=Meta({"scope": "ok"}),
        ),
    )
    async with ctx.scope("test"):
        assert await repository.load(Template.of("valid")) == "Hello {%name%}"


@pytest.mark.asyncio
async def test_file_repository_uses_empty_state_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    path.write_text("{not json", encoding="utf-8")
    repository = TemplatesRepository.file(path)

    async with ctx.scope("test"):
        assert (await repository.templates()).items == ()
        with pytest.raises(TemplateMissing, match="Missing template - missing"):
            await repository.load(Template.of("missing"))


@pytest.mark.asyncio
async def test_volatile_repository_templates_support_pagination() -> None:
    repository = TemplatesRepository.volatile(
        first="Hello",
        second="Hi",
        third="Hey",
    )

    page_1 = await repository.templates(Pagination.of(limit=2))

    assert tuple(template.identifier for template in page_1.items) == ("first", "second")
    assert page_1.pagination.token == "templates:2"

    page_2 = await repository.templates(page_1.pagination)

    assert tuple(template.identifier for template in page_2.items) == ("third",)
    assert page_2.pagination.token is None


@pytest.mark.asyncio
async def test_file_repository_templates_support_pagination(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    repository = TemplatesRepository.file(path)

    async with ctx.scope("test"):
        await repository.define(
            TemplateDeclaration.of("first"),
            content="Hello",
        )
        await repository.define(
            TemplateDeclaration.of("second"),
            content="Hi",
        )
        await repository.define(
            TemplateDeclaration.of("third"),
            content="Hey",
        )

        page_1 = await repository.templates(Pagination.of(limit=2))
        page_2 = await repository.templates(page_1.pagination)

    assert tuple(template.identifier for template in page_1.items) == ("first", "second")
    assert page_1.pagination.token == "templates:2"
    assert tuple(template.identifier for template in page_2.items) == ("third",)
    assert page_2.pagination.token is None


@pytest.mark.asyncio
async def test_resolve_preserves_multimodal_argument_metadata() -> None:
    repository = TemplatesRepository.volatile(
        welcome="Hello {%name%}!",
    )

    result = await repository.resolve(
        Template.of(
            "welcome",
            arguments={"name": TextContent.of("Ada", meta={"tone": "warm"})},
        )
    )

    assert len(result.parts) == 3
    assert result.parts[0] == TextContent.of("Hello ")
    assert result.parts[1] == TextContent.of("Ada", meta={"tone": "warm"})
    assert result.parts[2] == TextContent.of("!")


@pytest.mark.asyncio
async def test_resolve_loaded_template_eagerly_resolves_unused_template_argument() -> None:
    repository = TemplatesRepository.volatile(
        welcome="Hello {%name%}",
    )

    with pytest.raises(TemplateMissing, match="Missing template - missing"):
        await repository.resolve(
            Template.of("welcome", arguments={"name": "Ada"}),
            arguments={"unused": Template.of("missing")},
        )


@pytest.mark.asyncio
async def test_resolve_str_loaded_template_eagerly_resolves_unused_template_argument() -> None:
    repository = TemplatesRepository.volatile(
        welcome="Hello {%name%}",
    )

    with pytest.raises(TemplateMissing, match="Missing template - missing"):
        await repository.resolve_str(
            Template.of("welcome", arguments={"name": "Ada"}),
            arguments={"unused": Template.of("missing")},
        )


@pytest.mark.asyncio
async def test_resolve_raw_and_loaded_empty_templates_match() -> None:
    repository = TemplatesRepository.volatile(
        empty="",
    )

    raw = await repository.resolve("")
    loaded = await repository.resolve(Template.of("empty"))

    assert raw is MultimodalContent.empty
    assert loaded is MultimodalContent.empty
