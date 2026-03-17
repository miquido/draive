# Multimodal Content

`MultimodalContent` is the core immutable container for text, resources, artifacts, and tags.

It is used across generation input/output, tool payloads, and model context blocks.

## Quick Start

```python
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceContent

report = MultimodalContent.of(
    "Quarterly report:",
    ResourceContent.of(image_bytes, mime_type="image/png"),
    TextContent.of("Highlights: revenue up 18%"),
)
```

`MultimodalContent.of(...)` normalizes nested content and keeps ordering deterministic.

## Part Types

- `TextContent` for plain text payloads
- `ResourceContent` for inlined bytes (image/audio/video/documents)
- `ResourceReference` for URI-based external resources
- `ArtifactContent` for typed structured payloads
- `MultimodalTag` for lightweight XML-style tagging

## Typed Artifacts

Artifacts are ideal for internal typed payload exchange between workflow stages.

```python
from collections.abc import Sequence

from draive import State
from draive.multimodal import ArtifactContent, MultimodalContent


class ProductSummary(State, serializable=True):
    name: str
    price: float
    highlights: Sequence[str]


content = MultimodalContent.of(
    "Product spotlight",
    ArtifactContent.of(
        ProductSummary(name="Aurora Lamp", price=89.0, highlights=("Warm light", "USB-C")),
        category="product",
    ),
)
```

## Common Helpers

```python
# Metadata-based extraction and grouping.
user_parts = content.matching_meta(source="user")
section_groups = content.split_by_meta(key="section")

# Media and artifact extraction.
images = content.images()
resources = content.resources(mime_type="application/pdf")
artifacts = content.artifacts(model=ProductSummary, category="product")

# Build filtered variant for downstream processing.
clean = content.without_resources().without_artifacts()
```

## Tags

Use tags to wrap semantic regions while keeping them in multimodal flow.

```python
from draive.multimodal import MultimodalTag


title = MultimodalTag.of("Q1 Sales Report", name="title", meta={"lang": "en"})
document = MultimodalContent.of(title, "Summary: sales increased by 15%")

first_title = document.tag("title")
all_titles = document.tags("title")
```

## Using With Generation

```python
from draive import TextGeneration, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


async def analyze_image(image_bytes: bytes) -> str:
    async with ctx.scope(
        "image_analysis",
        OpenAIResponsesConfig(model="gpt-5"),
        disposables=(OpenAI(),),
    ):
        prompt = MultimodalContent.of(
            "Describe this image.",
            ResourceContent.of(image_bytes, mime_type="image/jpeg"),
        )
        return await TextGeneration.generate(
            instructions="Provide a concise analysis.",
            input=prompt,
        )
```

The same content helpers can be applied to outputs, which makes post-processing and auditing
consistent.
