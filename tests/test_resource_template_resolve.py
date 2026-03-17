from typing import Annotated

import pytest
from haiway import Alias

from draive.resources.template import resource
from draive.resources.types import ResourceContent, ResourceCorrupted


@pytest.mark.asyncio
async def test_resolve_expands_path_template() -> None:
    @resource(uri_template="https://api.example.com/users/{user_id}")
    async def get_user(user_id: str) -> ResourceContent:
        return ResourceContent.of(f"user:{user_id}".encode(), mime_type="text/plain")

    resolved = await get_user.resolve(user_id="123")

    assert resolved.uri == "https://api.example.com/users/123"
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.mime_type == "text/plain"


@pytest.mark.asyncio
async def test_resolve_from_uri_supports_alias_parameter() -> None:
    @resource(uri_template="https://api.example.com/{lang}")
    async def translate(language: Annotated[str, Alias("lang")]) -> ResourceContent:
        return ResourceContent.of(language.encode(), mime_type="text/plain")

    resolved = await translate.resolve_from_uri("https://api.example.com/pl")

    assert resolved.uri == "https://api.example.com/pl"
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"pl"


@pytest.mark.asyncio
async def test_resolve_from_uri_uses_default_for_optional_query_param() -> None:
    @resource(uri_template="https://api.example.com/search{?q,limit}")
    async def search(
        q: str,
        limit: int = 10,
    ) -> ResourceContent:
        return ResourceContent.of(f"{q}:{limit}".encode(), mime_type="text/plain")

    resolved = await search.resolve_from_uri("https://api.example.com/search?q=python")

    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"python:10"


@pytest.mark.asyncio
async def test_resolve_from_uri_raises_for_mismatched_uri() -> None:
    @resource(uri_template="https://api.example.com/users/{user_id}")
    async def get_user(user_id: str) -> ResourceContent:
        return ResourceContent.of(user_id.encode(), mime_type="text/plain")

    with pytest.raises(ResourceCorrupted):
        await get_user.resolve_from_uri("https://api.example.com/projects/1")
