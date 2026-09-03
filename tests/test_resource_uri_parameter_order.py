import pytest

from draive.resources import resource
from draive.resources.types import ResourceContent


@resource(uri_template="https://api.example.com/{org}{/repo}/tree")
async def mixed_before(org: str, repo: str) -> ResourceContent:
    return ResourceContent.of(f"org={org} repo={repo}".encode(), mime_type="text/plain")


@resource(uri_template="https://api.example.com{/repo}/{org}/tree")
async def mixed_after(org: str, repo: str) -> ResourceContent:
    return ResourceContent.of(f"org={org} repo={repo}".encode(), mime_type="text/plain")


@resource(uri_template="https://api.example.com/{a}{/b}/{c}{/d}/end")
async def interleaved(a: str, b: str, c: str, d: str) -> ResourceContent:
    return ResourceContent.of(f"{a}|{b}|{c}|{d}".encode(), mime_type="text/plain")


@pytest.mark.asyncio
async def test_slash_parameter_after_regular_keeps_order() -> None:
    resolved = await mixed_before.resolve_from_uri("https://api.example.com/acme/lib/tree")
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"org=acme repo=lib"


@pytest.mark.asyncio
async def test_slash_parameter_before_regular_keeps_order() -> None:
    resolved = await mixed_after.resolve_from_uri("https://api.example.com/lib/acme/tree")
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"org=acme repo=lib"


@pytest.mark.asyncio
async def test_interleaved_parameters_keep_order() -> None:
    resolved = await interleaved.resolve_from_uri("https://api.example.com/1/2/3/4/end")
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"1|2|3|4"


@pytest.mark.asyncio
async def test_literal_template_segments_are_matched_literally() -> None:
    # a literal `.` must not act as a regex wildcard
    assert not mixed_before.matches_uri("https://api.example.com/acme/lib/tre3")
    resolved = await mixed_before.resolve_from_uri("https://api.example.com/a.c/lib/tree")
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"org=a.c repo=lib"
