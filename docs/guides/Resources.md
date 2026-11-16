# Resources

Resources expose content that is addressed by a URI. A **resource template** declares the URI as an
[RFC 6570](https://www.rfc-editor.org/rfc/rfc6570) template and resolves its content with an async
function. Templates are backed by `URITemplate`, shared with Haiway.

## Declaring a resource template

```python
from draive import ResourceContent, ctx, resource


@resource(uri_template="https://api.example.com/{org}/{repo}/issues/{issue_id}{?state}")
async def issue(org: str, repo: str, issue_id: int, state: str = "open") -> ResourceContent:
    return ResourceContent.of(
        f"{org}/{repo}#{issue_id} ({state})".encode(),
        mime_type="text/plain",
    )


async def main() -> None:
    async with ctx.scope("resources"):
        resolved = await issue.resolve_from_uri(
            "https://api.example.com/miquido/draive/issues/42?state=closed"
        )
        print(resolved.uri)
```

Every template variable has to match a function argument by name or by `Alias`, otherwise the
declaration raises `ValueError`. Values taken from a URI are coerced to the annotated argument types
(`int`, `bool`, enums, JSON lists and objects, ...).

## URITemplate

`URITemplate` can be used directly and can also be passed to `@resource` in place of a string.

```python
from draive import URITemplate

template = URITemplate.of("https://{tenant}.example.com/files{/name}{?page}")

template.variables  # ("tenant", "name", "page")
template.expand({"tenant": "acme", "name": "a b.txt", "page": 2})
# "https://acme.example.com/files/a%20b.txt?page=2"
template.match("https://acme.example.com/files/a%20b.txt?page=2")
# {"tenant": "acme", "name": "a b.txt", "page": "2"}
```

Supported expressions are `{name}`, `{/name}`, `{?a,b}` (with `*` to explode lists) and `{#name}`. A
malformed template raises `ValidationError` when it is created.

## Matching rules

Matching is strict so templates can safely guard lookups:

- every path or host variable spans exactly one segment and is never empty,
- decoded values containing a path separator, a relative segment (`.`, `..`), a control character or
    (within the host) a delimiter or `@` do not match,
- a URI carrying query parameters the template does not declare is refused when the template
    declares no query at all; with a `{?...}` expression, undeclared parameters are ignored,
- expanding a template without a value for a required variable raises `ValueError`.

Use `ResourceTemplate.matches_uri(uri)` to check a URI without resolving it, and
`ResourceTemplate.resolve(**arguments)` to build the URI from arguments.
