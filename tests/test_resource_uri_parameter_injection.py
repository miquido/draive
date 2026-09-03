import pytest

from draive.resources import resource
from draive.resources.types import ResourceContent


@resource(uri_template="https://api.example.com/items/{item_id}")
async def undeclared_query(
    item_id: str,
    internal: bool = False,
    max_bytes: int = 1024,
    scopes: list[str] | None = None,
) -> ResourceContent:
    return ResourceContent.of(
        f"{item_id}|{internal}|{max_bytes}|{scopes}".encode(),
        mime_type="text/plain",
    )


@resource(uri_template="https://api.example.com/items/{item_id}{?limit}")
async def declared_query(
    item_id: str,
    limit: int = 10,
    internal: bool = False,
) -> ResourceContent:
    return ResourceContent.of(
        f"{item_id}|{limit}|{internal}".encode(),
        mime_type="text/plain",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "internal=true",
        "internal=yes&max_bytes=999999",
        'scopes=["admin","root"]',
        "max_bytes=1",
    ],
)
async def test_undeclared_query_parameters_are_not_bound(query: str) -> None:
    # the template exposes no query parameters, so defaults have to be preserved
    resolved = await undeclared_query.resolve_from_uri(f"https://api.example.com/items/abc?{query}")
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"abc|False|1024|None"


@pytest.mark.asyncio
async def test_declared_query_parameter_is_bound_while_others_are_not() -> None:
    resolved = await declared_query.resolve_from_uri(
        "https://api.example.com/items/abc?limit=5&internal=true"
    )
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"abc|5|False"


@pytest.mark.asyncio
async def test_declared_query_parameter_keeps_default_when_absent() -> None:
    resolved = await declared_query.resolve_from_uri("https://api.example.com/items/abc")
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"abc|10|False"


@pytest.mark.asyncio
async def test_undeclared_query_parameters_do_not_prevent_resolution() -> None:
    # unrelated query elements are ignored rather than rejected
    resolved = await declared_query.resolve_from_uri(
        "https://api.example.com/items/abc?limit=5&_cache=1234"
    )
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"abc|5|False"
