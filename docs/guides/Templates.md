# Templates

Templates provide a typed, multimodal-friendly replacement for the legacy instructions system. They
let you author reusable prompt fragments with parameter placeholders, resolve them at runtime, and
back them with any storage supported by `TemplatesRepository` (in-memory, file-backed, Postgres, or
your own adapter).

## Template Basics

```python
from draive import Template

welcome = Template.of(
    "welcome-email",
    arguments={"audience": "developers"},
)

personalised = welcome.with_arguments(product="Draive 2.0")
```

- `Template.of(...)` creates an immutable handle identified by `identifier`.
- `arguments` holds default values for `{% placeholders %}` embedded in the template source.
- Use `.with_arguments(...)` to merge additional arguments without mutating the original object.

Templates support multimodal values, so an argument can be plain text, `MultimodalContent`, or any
other part accepted by `MultimodalContent.of(...)`. When rendered, placeholders keep the modality of
the argument.

## Storing Templates

`TemplatesRepository` is the state that knows how to list, load, and define templates. You can pick
a storage backend depending on your workflow:

```python
from pathlib import Path
from draive import TemplatesRepository

file_repository = TemplatesRepository.file(Path("templates"))
volatile_repository = TemplatesRepository.volatile(
    onboarding="Hello {% user %}!",
)
```

- `TemplatesRepository.file(...)` reads templates from `.tmpl` files on disk. It automatically
  infers variables by scanning for `{% variable %}` markers.
- `TemplatesRepository.volatile(...)` keeps definitions in memory, ideal for tests or quick demos.
- Custom backends only need to provide the `listing`, `loading`, and `defining` callables. See
  `PostgresTemplatesRepository` for a production-ready example.

After constructing the repository, make it available in your Haiway context:

```python
from draive import ctx

async with ctx.scope(
    "demo",
    file_repository,
):
    ...
```

Any coroutine running inside that scope can now resolve templates through the active repository
state.

## Resolving Templates at Runtime

```python
from draive import TemplatesRepository

async def render_welcome(user_name: str) -> str:
    template = Template.of("welcome-email").with_arguments(user=user_name)
    return await TemplatesRepository.resolve_str(template)
```

- `resolve(...)` returns a `MultimodalContent` instance, keeping non-text arguments intact.
- `resolve_str(...)` flattens everything into text, useful for providers that only understand text.
- Pass `default="..."` to fall back to inline content when the template is missing in storage.
- If neither the storage nor `default` can satisfy a request, `TemplateMissing` is raised.

You can override argument values call-by-call:

```python
await TemplatesRepository.resolve_str(
    template,
    arguments={"cta": "Join the beta"},
)
```

Custom arguments are merged on top of any defaults stored in the `Template` instance.

## Listing and Managing Templates

```python
declarations = await TemplatesRepository.templates()
for declaration in declarations:
    print(declaration.identifier, declaration.variables)
```

- `templates()` returns `TemplateDeclaration` objects containing the identifier, optional
  description, discovered variables, and metadata.
- `TemplatesRepository.define(...)` (available on custom backends) persists a new revision and
  invalidates caches. File/volatile repositories expose it automatically through the state.

When defining templates programmatically, pass `variables={"user": "User name"}` to document
expected arguments. This metadata is surfaced in listings and downstream tooling.

## Migrating from InstructionsRepository

`TemplatesRepository` fully supersedes the deprecated `InstructionsRepository`. When updating older
code:

- Replace instruction names with template identifiers (`InstructionDeclaration` →
  `TemplateDeclaration`).
- Swap `InstructionsRepository.resolve(...)` with `TemplatesRepository.resolve_str(...)` or
  `resolve(...)` if you now need multimodal payloads.
- Update placeholders from legacy `{{ variable }}` markers to `{% variable %}`. The new syntax
  distinguishes literal braces from arguments and supports multimodal values.
- Remove instruction-specific argument lists; template arguments are simple mappings keyed by the
  placeholder name.

Combining Templates with `PostgresTemplatesRepository` or other storage adapters gives you revision
history, cache controls, and shared access across services while keeping the runtime API consistent.
