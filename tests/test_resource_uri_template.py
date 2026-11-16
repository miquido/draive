from typing import Annotated

import pytest
from haiway import Alias

from draive.resources.template import resource
from draive.resources.types import ResourceContent, ResourceCorrupted


@pytest.mark.asyncio
async def test_resolve_from_uri_with_path_parameters() -> None:
    @resource(uri_template="https://api.example.com/{org}/{repo}/issues/{issue_id}")
    async def issue(org: str, repo: str, issue_id: str) -> ResourceContent:
        return ResourceContent.of(f"{org}/{repo}#{issue_id}".encode(), mime_type="text/plain")

    resolved = await issue.resolve_from_uri("https://api.example.com/octocat/hello-world/issues/42")

    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"octocat/hello-world#42"


@pytest.mark.asyncio
async def test_resolve_from_uri_with_query_parameters() -> None:
    @resource(uri_template="https://api.example.com/search{?q,limit}")
    async def search(q: str, limit: int) -> ResourceContent:
        return ResourceContent.of(f"{q}:{limit}".encode(), mime_type="text/plain")

    resolved = await search.resolve_from_uri("https://api.example.com/search?q=python&limit=20")

    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"python:20"


@pytest.mark.asyncio
async def test_resolve_from_uri_with_alias_query_parameter() -> None:
    @resource(uri_template="https://api.example.com/search{?language}")
    async def search(language: Annotated[str, Alias("lang")]) -> ResourceContent:
        return ResourceContent.of(language.encode(), mime_type="text/plain")

    resolved = await search.resolve_from_uri("https://api.example.com/search?language=pl")

    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"pl"


@pytest.mark.asyncio
async def test_resolve_from_uri_raises_for_invalid_uri() -> None:
    @resource(uri_template="https://api.example.com/status")
    async def status() -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    with pytest.raises(ResourceCorrupted):
        await status.resolve_from_uri("https://api.example.com/status?unexpected=true")


@pytest.mark.asyncio
async def test_resource_accepts_a_prepared_uri_template() -> None:
    from haiway import URITemplate

    template = URITemplate.of("https://api.example.com/users/{user_id}{?verbose}")

    @resource(uri_template=template)
    async def user(user_id: str, verbose: bool = False) -> ResourceContent:
        return ResourceContent.of(f"{user_id}|{verbose}".encode(), mime_type="text/plain")

    assert user.declaration.template_uri == template
    assert user.has_args
    assert (
        await user.resolve(user_id="a b", verbose=True)
    ).uri == "https://api.example.com/users/a%20b?verbose=true"


def test_resource_rejects_a_malformed_uri_template() -> None:
    from haiway import ValidationError

    with pytest.raises(ValidationError):

        @resource(uri_template="https://api.example.com/{user_id")
        async def broken(user_id: str) -> ResourceContent:
            return ResourceContent.of(b"", mime_type="text/plain")


def test_resource_without_variables_has_no_args() -> None:
    @resource(uri_template="https://api.example.com/status")
    async def status() -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    assert not status.has_args
    assert status.matches_uri("https://api.example.com/status")
    assert not status.matches_uri("https://api.example.com/other")


@pytest.mark.asyncio
async def test_resolve_expands_aliased_template_variable() -> None:
    @resource(uri_template="https://api.example.com/search{?lang}")
    async def search(language: Annotated[str, Alias("lang")]) -> ResourceContent:
        return ResourceContent.of(language.encode(), mime_type="text/plain")

    assert (await search.resolve(language="pl")).uri == "https://api.example.com/search?lang=pl"
