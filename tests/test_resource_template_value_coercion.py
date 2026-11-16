from typing import Any, Literal

import pytest
from haiway import ctx

from draive.resources import ResourceContent, resource


def _content() -> ResourceContent:
    return ResourceContent.of(b"x", mime_type="text/plain")


@pytest.mark.asyncio
async def test_literal_string_values_keep_their_type() -> None:
    captured: dict[str, Any] = {}

    async def read(mode: Literal["a", "1"]) -> ResourceContent:
        captured["mode"] = mode
        return _content()

    async with ctx.scope("test"):
        await resource(uri_template="res://{mode}")(read).resolve_from_uri("res://1")

    assert captured["mode"] == "1"


@pytest.mark.asyncio
async def test_plain_string_parameter_is_never_converted() -> None:
    captured: dict[str, Any] = {}

    async def read(query: str) -> ResourceContent:
        captured["query"] = query
        return _content()

    async with ctx.scope("test"):
        await resource(uri_template="res://{query}")(read).resolve_from_uri("res://123")

    assert captured["query"] == "123"


@pytest.mark.asyncio
async def test_unicode_parameter_round_trips_through_uri() -> None:
    captured: dict[str, Any] = {}

    async def read(query: str) -> ResourceContent:
        captured["query"] = query
        return _content()

    template = resource(uri_template="res://{query}/doc")(read)
    async with ctx.scope("test"):
        resolved = await template.resolve(query="日本語é")

        assert template.matches_uri(resolved.uri)
        await template.resolve_from_uri(resolved.uri)

    assert captured["query"] == "日本語é"


@pytest.mark.asyncio
async def test_query_parameters_fall_back_to_defaults_and_ignore_unknown() -> None:
    captured: dict[str, Any] = {}

    async def read(query: str | None = None, other: str = "default") -> ResourceContent:
        captured["values"] = (query, other)
        return _content()

    template = resource(uri_template="res://doc{?query,other}")(read)
    async with ctx.scope("test"):
        await template.resolve_from_uri("res://doc?query=%E6%97%A5&unknown=1")

    assert captured["values"] == ("日", "default")


@pytest.mark.asyncio
@pytest.mark.parametrize("literal", ("null", "None"))
async def test_optional_string_parameter_keeps_null_like_text(literal: str) -> None:
    captured: dict[str, Any] = {}

    async def read(query: str | None = None) -> ResourceContent:
        captured["query"] = query
        return _content()

    async with ctx.scope("test"):
        await resource(uri_template="res://{query}")(read).resolve_from_uri(f"res://{literal}")

    assert captured["query"] == literal


@pytest.mark.asyncio
async def test_string_or_int_parameter_prefers_string() -> None:
    captured: dict[str, Any] = {}

    async def read(query: str | int) -> ResourceContent:
        captured["query"] = query
        return _content()

    async with ctx.scope("test"):
        await resource(uri_template="res://{query}")(read).resolve_from_uri("res://007")

    assert captured["query"] == "007"


@pytest.mark.asyncio
async def test_optional_int_parameter_still_coerces_null_and_numbers() -> None:
    captured: list[Any] = []

    async def read(query: int | None = 1) -> ResourceContent:
        captured.append(query)
        return _content()

    template = resource(uri_template="res://{query}")(read)
    async with ctx.scope("test"):
        await template.resolve_from_uri("res://null")
        await template.resolve_from_uri("res://007")

    assert captured == [None, 7]
