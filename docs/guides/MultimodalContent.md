# Multimodal Content

`MultimodalContent` is the core container for mixing text, media, and structured data across Draive
pipelines. It keeps parts immutable, preserves ordering, and exposes rich helpers to query and
transform what you feed into or receive from models.

## Quick Start

```python
from draive import MultimodalContent, TextContent, ResourceContent

report = MultimodalContent.of(
    "Quarterly report:",
    ResourceContent.of(image_bytes, mime_type="image/png"),
    TextContent.of("Highlights: revenue up 18%"),
)
```

- Input accepts plain strings, `TextContent`, `ResourceContent`, `ResourceReference`,
  `ArtifactContent`, other `MultimodalContent`, and `MultimodalTag` instances.
- Construction is normalized: adjacent text fragments are merged and existing `MultimodalContent`
  instances are reused.
- The resulting object is immutable; every mutating-style call returns a new instance.

## Building Blocks

| Part type           | When to use                                                             | Key helpers                                   |
| ------------------- | ----------------------------------------------------------------------- | --------------------------------------------- |
| `TextContent`       | Text blocks with optional metadata (e.g. author, language)              | `.of(...)`, `.meta`, `.to_str()`              |
| `ResourceContent`   | Inline binary data such as images or audio                              | `.of(bytes, mime_type=...)`, `.to_data_uri()` |
| `ResourceReference` | Reference external resources by URI when you cannot embed bytes         | `.of(url, mime_type=...)`                     |
| `ArtifactContent`   | Wrap strongly typed `DataModel` objects for passing structured payloads | `.of(artifact, category=...)`                 |
| `MultimodalTag`     | Lightweight XML-like tags for templating or post-processing             | `.of(content, name=..., meta=...)`            |

All parts share the same metadata model, so you can attach consistent descriptors like
`{"section": "summary"}` regardless of part type.

## Creating Content

```python
from collections.abc import Sequence
from draive import DataModel, MultimodalContent, ArtifactContent, ResourceReference, TextContent

class ProductSummary(DataModel):
    name: str
    price: float
    highlights: Sequence[str]

thumbnail = ResourceReference.of(
    "https://cdn.example.com/items/1234/thumbnail.jpg",
    mime_type="image/jpeg",
)

content = MultimodalContent.of(
    TextContent.of("Product spotlight", meta={"section": "title"}),
    thumbnail,
    ArtifactContent.of(
        ProductSummary(
            name="Aurora Lamp",
            price=89.0,
            highlights=("Warm light", "USB-C power"),
        ),
        category="product",
    ),
)
```

## Metadata & Organization

Use metadata to organize parts and later filter or group them.

```python
from draive import MultimodalContent

# Filter by exact metadata match
user_parts = content.matching_meta(source="user", section="summary")

# Group whenever a meta key changes value
section_groups = content.split_by_meta(key="section")
```

Common patterns:

- Tag user-generated vs. system-generated text with `{ "source": "user" }` and
  `{ "source": "assistant" }`.
- Store structured routing hints like `{ "stage": "retrieval" }` to drive downstream stages.

## Working with Resources and Artifacts

```python
# Resource helpers
if content.contains_resources:
    images = content.images()
    audio_tracks = content.audio()
    video_clips = content.video()

resources = content.resources(mime_type="application/pdf")
content_without_media = content.without_resources()

# Artifact helpers
artifact_profiles = content.artifacts(model=ProductSummary)
content_without_artifacts = content.without_artifacts()
```

- `resources(mime_type=...)` narrows by MIME type (exact match).
- `artifacts(model=..., category=...)` lets you filter by wrapped `DataModel` type and logical
  category.

## Tagging & Lightweight Markup

Wrap content in `MultimodalTag` when you need template-like markers that survive round-trips through
generation.

```python
from draive import MultimodalContent, MultimodalTag, TextContent

title = MultimodalTag.of(
    TextContent.of("Q1 Sales Report"),
    name="title",
    meta={"lang": "en"},
)

document = MultimodalContent.of(
    title,
    "Summary: sales increased by 15%",
)

first_title = document.tag("title")
all_titles = document.tags("title")

updated = document.replacing_tag(
    "title",
    MultimodalContent.of("Updated Title"),
    strip_tags=True,
)
```

- `tag(name)` returns the first matching tag; `tags(name)` returns all.
- `replacing_tag(...)` swaps one or all occurrences. Pass `strip_tags=True` to unwrap the original
  tag markers.

## Transformations & Utilities

```python
# Append new pieces
extended = content.appending("Additional notes", thumbnail)

# Combine multiple multimodal payloads
final = MultimodalContent.of(content, extended)

# Render a text-only view (resources become placeholders)
text_view = content.to_str()
```

Because every method returns a new instance, you can chain calls to build complex documents while
keeping the original inputs intact.

## Using in Generation Pipelines

```python
import asyncio
from draive import MultimodalContent, ResourceContent, TextGeneration, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig

async def analyze_image(image_bytes: bytes) -> str:
    async with ctx.scope(
        "image_analysis",
        OpenAIResponsesConfig(model="gpt-5"),
        disposables=(OpenAI(),),
    ):
        prompt = MultimodalContent.of(
            "Describe this image",
            ResourceContent.of(image_bytes, mime_type="image/jpeg"),
        )
        response = await TextGeneration.generate(
            instructions="Provide a concise analysis",
            input=prompt,
        )
        return response.to_str()

asyncio.run(analyze_image(b"..."))
```

This example scopes provider configuration, builds a mixed prompt, and feeds it directly into a
generation facade. The response is also a `MultimodalContent`, so you can reuse the same helpers
when handling model outputs.

## Best Practices

- Keep metadata small and serializable; prefer strings, numbers, and tuples of primitives.
- Embed only assets that must travel with the request. Use `ResourceReference` for large, cacheable
  files.
- Normalize user content early to attach provenance metadata and validate MIME types.
- When chaining transformations, retain the original content for auditing by storing both the source
  and derived instances.
- Use `ArtifactContent` to bridge typed internal data (e.g. summaries, retrieval chunks) rather than
  serializing to JSON manually.

With these patterns you can confidently build, inspect, and transform rich multimodal payloads while
preserving structure for downstream Draive stages.
