<!-- markdownlint-disable-file MD046 -->

# Multimodal Data

`Multimodal` is the normalized input/output type used by Draive generation APIs.

Instead of handling separate text/image/audio types at each boundary, you can pass one multimodal
value and let Draive normalize it.

It accepts plain text and typed multimodal parts such as:

- `TextContent`
- `ResourceContent` / `ResourceReference`
- `ArtifactContent`
- `MultimodalTag`
- `MultimodalContent` (already-composed payload)

## `MultimodalContent`

`MultimodalContent` is the main immutable container for multimodal parts.

```python
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceContent

content = MultimodalContent.of(
    TextContent.of("Describe the image"),
    ResourceContent.of(image_bytes, mime_type="image/jpeg"),
)
```

Useful helpers include:

- `texts()`, `images()`, `audio()`, `resources()` for part extraction
- `artifacts(...)`, `tags(...)` for structured component retrieval
- `matching_meta(...)`, `split_by_meta(...)` for metadata-aware filtering
- `without_resources()`, `without_artifacts()` for creating reduced variants

## Artifacts With Typed State

Use artifacts to move typed payloads through multimodal flows.

```python
from draive import State
from draive.multimodal import ArtifactContent, MultimodalContent


class User(State, serializable=True):
    first_name: str
    last_name: str


payload = MultimodalContent.of(
    "User profile",
    ArtifactContent.of(User(first_name="James", last_name="Smith"), category="profile"),
)
```

## Model Input/Output Context

`ModelInput` and `ModelOutput` store `MultimodalContent` blocks, so the same content model is used
for:

- direct user input,
- model responses,
- tool request/response payloads.

This keeps transformations, filtering, and observability logic consistent across pipeline stages.
