from pytest import raises

from draive.multimodal.content import MultimodalContent
from draive.multimodal.templates.variables import (
    parse_template_variables,
    resolve_multimodal_template,
    resolve_text_template,
)
from draive.multimodal.text import TextContent


def test_parse_template_variables_finds_unique_names() -> None:
    template = "Hello {%name%}, you have {%count%} new messages from {%name%}."

    variables = parse_template_variables(template)

    assert variables == {"name", "count"}


def test_parse_template_variables_without_placeholders_returns_empty() -> None:
    assert parse_template_variables("No placeholders here.") == set()


def test_parse_template_variables_skips_unmatched_closing() -> None:
    template = "Hello %} stray closing"

    variables = parse_template_variables(template)

    assert variables == set()


def test_parse_template_variables_skips_unclosed_opening() -> None:
    template = "Hello {%name"

    variables = parse_template_variables(template)

    assert variables == set()


def test_parse_template_variables_skips_empty_name() -> None:
    template = "Hello {% %}"

    variables = parse_template_variables(template)

    assert variables == set()


def test_parse_template_variables_skips_whitespace_in_name() -> None:
    template = "Hello {%bad name%}"

    variables = parse_template_variables(template)

    assert variables == set()


def test_parse_template_variables_skips_padded_name() -> None:
    template = "Hello {% name %}"

    variables = parse_template_variables(template)

    assert variables == set()


def test_parse_template_variables_handles_adjacent_placeholders() -> None:
    template = "{%first%}{%second%}"

    variables = parse_template_variables(template)

    assert variables == {"first", "second"}


def test_parse_template_variables_supports_symbolic_names() -> None:
    template = "{%section-2%}"

    variables = parse_template_variables(template)

    assert variables == {"section-2"}


def test_parse_template_variables_handles_literal_braces() -> None:
    template = "{{ not a placeholder }}"

    variables = parse_template_variables(template)

    assert variables == set()


def test_resolve_text_template_replaces_all_placeholders() -> None:
    template = "Hi {%name%}, welcome to {%place%}."
    arguments = {"name": "Ada", "place": "Paris"}

    result = resolve_text_template(template, arguments=arguments)

    assert result == "Hi Ada, welcome to Paris."


def test_resolve_text_template_supports_repeated_placeholder() -> None:
    template = "{%word%}-{%word%}-{%word%}"
    arguments = {"word": "repeat"}

    result = resolve_text_template(template, arguments=arguments)

    assert result == "repeat-repeat-repeat"


def test_resolve_text_template_detects_missing_argument() -> None:
    with raises(KeyError, match="Missing template argument: name"):
        resolve_text_template("Hello {%name%}", arguments={"other": "value"})

def test_resolve_text_template_ignores_unused_argument() -> None:
    result = resolve_text_template("Hello world", arguments={"extra": "value"})

    assert result == "Hello world"


def test_resolve_text_template_resolves_edge_positions() -> None:
    template = "{%greeting%}, middle, {%closing%}"
    arguments = {"greeting": "Hi", "closing": "Bye"}

    result = resolve_text_template(template, arguments=arguments)

    assert result == "Hi, middle, Bye"


def test_resolve_text_template_handles_empty_template() -> None:
    assert resolve_text_template("", arguments={}) == ""


def test_resolve_multimodal_template_concatenates_parts() -> None:
    template = "Hello {%name%}!"
    arguments = {"name": TextContent.of("world")}

    result = resolve_multimodal_template(template, arguments=arguments)

    assert isinstance(result, MultimodalContent)
    assert result.to_str() == "Hello world!"


def test_resolve_multimodal_template_accepts_string_parts() -> None:
    template = "{%greeting%} {%subject%}"
    arguments = {
        "greeting": "Hello",
        "subject": TextContent.of("Ada"),
    }

    result = resolve_multimodal_template(template, arguments=arguments)

    assert len(result.parts) == 1
    text_part = result.parts[0]
    assert isinstance(text_part, TextContent)
    assert text_part.text == "Hello Ada"


def test_resolve_multimodal_template_detects_missing_argument() -> None:
    with raises(KeyError, match="Missing template argument: name"):
        resolve_multimodal_template("Hello {%name%}", arguments={})


def test_resolve_multimodal_template_ignores_unused_argument() -> None:
    result = resolve_multimodal_template(
        "Plain text",
        arguments={"extra": TextContent.of("value")},
    )

    assert result.to_str() == "Plain text"


def test_resolve_multimodal_template_flattens_nested_content() -> None:
    template = "{%intro%} {%body%}"
    arguments = {
        "intro": MultimodalContent.of(TextContent.of("Hello"), TextContent.of(" there")),
        "body": TextContent.of("reader"),
    }

    result = resolve_multimodal_template(template, arguments=arguments)

    assert result.to_str() == "Hello there reader"


def test_resolve_multimodal_template_preserves_metadata_boundaries() -> None:
    template = "Hi {%name%}!"
    arguments = {
        "name": TextContent.of("Ada", meta={"tone": "warm"}),
    }

    result = resolve_multimodal_template(template, arguments=arguments)

    assert len(result.parts) == 3
    hi_part, name_part, exclaim_part = result.parts

    assert isinstance(hi_part, TextContent)
    assert hi_part.text == "Hi "
    assert hi_part.meta == {}

    assert isinstance(name_part, TextContent)
    assert name_part.text == "Ada"
    assert name_part.meta == {"tone": "warm"}

    assert isinstance(exclaim_part, TextContent)
    assert exclaim_part.text == "!"
    assert exclaim_part.meta == {}


def test_resolve_multimodal_template_returns_empty_content_for_empty_template() -> None:
    result = resolve_multimodal_template("", arguments={})

    assert result is MultimodalContent.empty
