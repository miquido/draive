import json
from typing import Annotated

from haiway import Alias, Default
from pytest import mark

from draive.resources.template import resource
from draive.resources.types import ResourceContent


@mark.asyncio
async def test_resolve_simple_path_expansion():
    @resource(uri_template="https://api.example.com/users/{user_id}")
    async def get_user(user_id: str) -> ResourceContent:
        return ResourceContent(
            data=json.dumps({"id": user_id}).encode().decode(),
            mime_type="application/json",
        )

    res = await get_user.resolve(user_id="123")
    assert res.uri == "https://api.example.com/users/123"
    assert res.meta.name == "get_user"


@mark.asyncio
async def test_resolve_query_includes_only_provided():
    @resource(uri_template="https://api.example.com/search{?q,limit}")
    async def search(q: str, limit: int = 10) -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    res = await search.resolve(q="python")
    assert res.uri == "https://api.example.com/search?q=python"


@mark.asyncio
async def test_resolve_slash_and_id_segments():
    @resource(uri_template="https://api.example.com{/resource}/{resource_id}")
    async def fetch(resource: str, resource_id: int) -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    res = await fetch.resolve(resource="users", resource_id=123)
    assert res.uri == "https://api.example.com/users/123"


@mark.asyncio
async def test_resolve_fragment_only_when_provided():
    @resource(uri_template="https://api.example.com/doc{#section}")
    async def doc(section: str | None = None) -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    res = await doc.resolve(section="intro")
    assert res.uri == "https://api.example.com/doc#intro"

    res2 = await doc.resolve()
    assert res2.uri == "https://api.example.com/doc"


@mark.asyncio
async def test_resolve_encoding_of_values():
    @resource(uri_template="https://api.example.com/items/{item_id}{?q}")
    async def items(item_id: str, q: str) -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    res = await items.resolve(item_id="a/b c", q="python rocks")
    # path encodes '/' and space; query encodes space as '+' via urlencode
    assert res.uri == "https://api.example.com/items/a%2Fb%20c?q=python+rocks"


@mark.asyncio
async def test_resolve_uses_canonical_name_for_aliased_argument():
    @resource(uri_template="https://api.example.com/search{?language}")
    async def search(language: Annotated[str, Alias("lang")]) -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    res = await search.resolve(**{"lang": "pl"})

    assert res.uri == "https://api.example.com/search?language=pl"


@mark.asyncio
async def test_template_parameter_accepts_alias_in_declaration():
    @resource(uri_template="https://api.example.com/{lang}")
    async def translate(language: Annotated[str, Alias("lang")]) -> ResourceContent:
        return ResourceContent.of(b"ok", mime_type="text/plain")

    res = await translate.resolve_from_uri("https://api.example.com/en")

    assert res.uri == "https://api.example.com/en"
    assert res.content.mime_type == "text/plain"


@mark.asyncio
async def test_resolve_from_uri_prefers_provided_values_without_calling_default():
    default_calls: list[int] = [0]

    def language_default() -> str:
        default_calls[0] += 1
        return "fallback"

    @resource(uri_template="https://api.example.com/{lang}")
    async def translate(
        language: Annotated[str, Alias("lang")] = Default(default_factory=language_default),
    ) -> ResourceContent:
        return ResourceContent.of(language.encode(), mime_type="text/plain")

    res = await translate.resolve_from_uri("https://api.example.com/pl")

    assert res.uri == "https://api.example.com/pl"
    assert default_calls[0] == 0
    assert res.content.mime_type == "text/plain"


@mark.asyncio
async def test_resolve_from_uri_uses_canonical_name_before_default():
    default_calls: list[int] = [0]

    def language_default() -> str:
        default_calls[0] += 1
        return "fallback"

    @resource(uri_template="https://api.example.com/{language}")
    async def translate(
        language: Annotated[str, Alias("lang")] = Default(default_factory=language_default),
    ) -> ResourceContent:
        return ResourceContent.of(language.encode(), mime_type="text/plain")

    res = await translate.resolve_from_uri("https://api.example.com/de")

    assert res.uri == "https://api.example.com/de"
    assert default_calls[0] == 0
    assert res.content.mime_type == "text/plain"
