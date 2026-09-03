import pytest

from draive.resources import resource
from draive.resources.types import ResourceContent, ResourceCorrupted


@resource(uri_template="https://{tenant}.example.com{/repo}")
async def tenant_slash_resource(tenant: str, repo: str) -> ResourceContent:
    return ResourceContent.of(f"{tenant}|{repo}".encode(), mime_type="text/plain")


@resource(uri_template="https://{tenant}.example.com/repos/{repo}")
async def tenant_path_resource(
    tenant: str = "default",
    repo: str = "default",
) -> ResourceContent:
    return ResourceContent.of(f"{tenant}|{repo}".encode(), mime_type="text/plain")


@resource(uri_template="https://{tenant}.example.com{/repo}{?ref}")
async def tenant_query_resource(
    tenant: str,
    repo: str,
    ref: str = "main",
) -> ResourceContent:
    return ResourceContent.of(f"{tenant}|{repo}|{ref}".encode(), mime_type="text/plain")


@pytest.mark.asyncio
async def test_authority_variable_is_extracted_with_slash_parameter() -> None:
    uri: str = "https://acme.example.com/lib"
    assert tenant_slash_resource.matches_uri(uri)
    resolved = await tenant_slash_resource.resolve_from_uri(uri)
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"acme|lib"


@pytest.mark.asyncio
async def test_authority_variable_is_extracted_with_path_parameter() -> None:
    # a matched URI must never fall back to the argument defaults, silently serving
    # a different tenant than the one it names
    uri: str = "https://acme.example.com/repos/lib"
    assert tenant_path_resource.matches_uri(uri)
    resolved = await tenant_path_resource.resolve_from_uri(uri)
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"acme|lib"


@pytest.mark.asyncio
async def test_authority_variable_is_extracted_alongside_query_parameter() -> None:
    uri: str = "https://acme.example.com/lib?ref=dev"
    assert tenant_query_resource.matches_uri(uri)
    resolved = await tenant_query_resource.resolve_from_uri(uri)
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"acme|lib|dev"


@pytest.mark.asyncio
async def test_authority_variable_expands_back_to_the_uri() -> None:
    resolved = await tenant_slash_resource.resolve(tenant="acme", repo="lib")
    assert resolved.uri == "https://acme.example.com/lib"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/lib",  # authority variable can't be empty
        "https://acme.other.com/lib",  # authority literal has to match
        "https://acme.example.com/nested/lib",  # {/var} spans a single segment
        "https://acme.example.com",  # missing path segment
    ],
)
async def test_authority_template_rejects_other_uris(uri: str) -> None:
    assert not tenant_slash_resource.matches_uri(uri)
    with pytest.raises(ResourceCorrupted):
        await tenant_slash_resource.resolve_from_uri(uri)


@pytest.mark.asyncio
async def test_authority_variable_rejects_encoded_separators() -> None:
    # the authority variable spans a single label group, decoding must not escape it
    uri: str = "https://acme%2Fsecret.example.com/lib"
    assert tenant_slash_resource.matches_uri(uri)
    with pytest.raises(ResourceCorrupted):
        await tenant_slash_resource.resolve_from_uri(uri)


@resource(uri_template="https://example.com/users/{account}")
async def account_path_resource(account: str) -> ResourceContent:
    return ResourceContent.of(account.encode(), mime_type="text/plain")


@pytest.mark.asyncio
async def test_authority_variable_rejects_raw_user_info() -> None:
    # `evil.com@acme.example.com` addresses the host `acme.example.com`, the authority
    # variable must not swallow the user-info and stand in for a host label it never was
    uri: str = "https://evil.com@acme.example.com/lib"
    assert not tenant_slash_resource.matches_uri(uri)
    with pytest.raises(ResourceCorrupted):
        await tenant_slash_resource.resolve_from_uri(uri)


@pytest.mark.asyncio
async def test_authority_variable_rejects_raw_user_info_with_password() -> None:
    uri: str = "https://user:pass@acme.example.com/lib"
    assert not tenant_slash_resource.matches_uri(uri)
    with pytest.raises(ResourceCorrupted):
        await tenant_slash_resource.resolve_from_uri(uri)


@pytest.mark.asyncio
async def test_authority_variable_rejects_encoded_user_info() -> None:
    # the raw capture already excludes `@`, decoding `%40` must not reintroduce it
    uri: str = "https://evil.com%40acme.example.com/lib"
    assert tenant_slash_resource.matches_uri(uri)
    with pytest.raises(ResourceCorrupted):
        await tenant_slash_resource.resolve_from_uri(uri)


@pytest.mark.asyncio
async def test_authority_variable_within_path_template_rejects_user_info() -> None:
    for uri in (
        "https://evil.com@acme.example.com/repos/lib",
        "https://evil.com%40acme.example.com/repos/lib",
    ):
        with pytest.raises(ResourceCorrupted):
            await tenant_path_resource.resolve_from_uri(uri)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://acme.example.com/some@repo", b"acme|some@repo"),
        ("https://acme.example.com/some%40repo", b"acme|some@repo"),
    ],
)
async def test_path_variable_keeps_accepting_user_info_delimiter(
    uri: str,
    expected: bytes,
) -> None:
    # only the authority is held to the narrower rule - `@` remains a valid path character
    assert tenant_slash_resource.matches_uri(uri)
    resolved = await tenant_slash_resource.resolve_from_uri(uri)
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/users/user@example.org",
        "https://example.com/users/user%40example.org",
    ],
)
async def test_path_variable_of_literal_authority_template_accepts_user_info(uri: str) -> None:
    assert account_path_resource.matches_uri(uri)
    resolved = await account_path_resource.resolve_from_uri(uri)
    assert isinstance(resolved.resource, ResourceContent)
    assert resolved.resource.to_bytes() == b"user@example.org"
