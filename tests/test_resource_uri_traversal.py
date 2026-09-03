import pytest

from draive.resources import resource
from draive.resources.types import ResourceContent, ResourceCorrupted


@resource(uri_template="file:///base/{filename}")
async def file_resource(filename: str) -> ResourceContent:
    return ResourceContent.of(filename.encode(), mime_type="text/plain")


@resource(uri_template="file:///base{/filename}")
async def slash_file_resource(filename: str) -> ResourceContent:
    return ResourceContent.of(filename.encode(), mime_type="text/plain")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded",
    [
        "..%2Fsecret.txt",  # encoded separator
        "%2e%2e%2fsecret.txt",  # encoded separator and dots
        "%2E%2E%5Csecret.txt",  # encoded backslash separator
        "%2e%2e",  # bare relative segment
        "%2e",  # bare current segment
        "nested%2Fsecret.txt",  # separator without relative segment
    ],
)
async def test_encoded_path_separators_do_not_resolve(encoded: str) -> None:
    # the pattern accepts a single segment, so decoding must not produce more than one
    for template in (file_resource, slash_file_resource):
        uri: str = f"file:///base/{encoded}"
        with pytest.raises(ResourceCorrupted):
            await template.resolve_from_uri(uri)


@pytest.mark.asyncio
async def test_literal_path_separators_do_not_match() -> None:
    uri: str = "file:///base/../secret.txt"
    assert not file_resource.matches_uri(uri)
    with pytest.raises(ResourceCorrupted):
        await file_resource.resolve_from_uri(uri)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [
        ("plain.txt", "plain.txt"),
        ("with%20space.txt", "with space.txt"),
        ("dots..in.name.txt", "dots..in.name.txt"),
        ("...", "..."),
        ("%3Fquestion.txt", "?question.txt"),
    ],
)
async def test_legitimate_encoded_segments_still_resolve(
    encoded: str,
    decoded: str,
) -> None:
    for template in (file_resource, slash_file_resource):
        resolved = await template.resolve_from_uri(f"file:///base/{encoded}")
        assert isinstance(resolved.resource, ResourceContent)
        assert resolved.resource.to_bytes() == decoded.encode()


@pytest.mark.asyncio
async def test_slash_template_does_not_match_multiple_segments() -> None:
    # {/var} expands to exactly one segment, so it has to match exactly one
    assert not slash_file_resource.matches_uri("file:///base/nested/secret.txt")
    with pytest.raises(ResourceCorrupted):
        await slash_file_resource.resolve_from_uri("file:///base/nested/secret.txt")
