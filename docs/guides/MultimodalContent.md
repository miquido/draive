# Multimodal Content


```python
from draive import MultimodalContent, TextContent, ResourceContent

# Create multimodal content from various sources
content = MultimodalContent.of(
    "Here's an image:",
    ResourceContent.of(image_bytes, mime_type="image/jpeg"),
    "And some additional text."


- Text: `TextContent` — plain text with optional metadata
- Resources: `ResourceContent` (embedded data) and `ResourceReference` (by URI)
- Artifacts: `ArtifactContent` — typed, structured data with category and metadata
- Tags: `MultimodalTag` — lightweight, XML-like tags wrapping content

```python
from draive import TextContent, ArtifactContent, DataModel, ResourceContent, ResourceReference

# Text content
text = TextContent.of("Hello world!")

# Resource data (embedded)
image_data = ResourceContent.of(image_bytes, mime_type="image/png")

# Resource reference (URL)
image_ref = ResourceReference.of(
    "https://example.com/image.jpg",
    mime_type="image/jpeg",
)

# Artifact content (structured data)
class UserProfile(DataModel):
    name: str
    age: int


from draive import MultimodalContent, TextContent

# From string
content = MultimodalContent.of("Simple text")

# From multiple elements
content = MultimodalContent.of(
    "Description:",
    image_data,
    "Additional context",
)

# Add metadata to specific parts
text_with_meta = TextContent.of("Labeled text", meta={"source": "user"})

# Check for resource presence
if content.contains_resources:
    print("Content contains resources")

# Get all resources or specific types
all_resources = content.resources()
images = content.images()
audio_files = content.audio()
videos = content.video()

# Access text parts
texts = content.texts()  # Sequence[TextContent]

# Remove resources


```python
from draive import DataModel, ArtifactContent

class UserProfile(DataModel):
    name: str
    age: int

# Content with artifacts
content = MultimodalContent.of(
    "User information:",
    ArtifactContent.of(UserProfile(name="John", age=30), category="profile"),
)

# Get all artifacts
all_artifacts = content.artifacts()

# Get specific artifact type
profiles = content.artifacts(UserProfile)

# Remove artifacts
clean_content = content.without_artifacts()
```


```python
from draive import TextContent
from draive import MultimodalTag

# Filter parts by exact metadata match
filtered = content.matching_meta(source="user", language="en")

# Split into groups where a meta key changes
groups = content.split_by_meta(key="section")

# Tagging: wrap content in a lightweight XML-like tag
title_tag = MultimodalTag.of(
    TextContent.of("Q1 Sales Report"),
    name="title",
    meta={"lang": "en"},
)
document = MultimodalContent.of(
    title_tag,
    "Summary: Sales increased by 15%",
)

# Find tags
first_title = document.tag("title")
all_titles = document.tags("title")

# Replace a tag with generated content (keep inner content)
updated = document.replacing_tag(
    "title",
    MultimodalContent.of("Updated Title"),
    strip_tags=True,

# Append new content
extended = content.appending(
    "Additional text",
    image_ref,
)

# Extend with other multimodal content
other_content = MultimodalContent.of("More content")

# Convert entire content to string
text_repr = content.to_str()  # Resource parts render as placeholders

# Obtain data URIs for embedded resources explicitly (ResourceContent only)
from draive import ResourceContent

from draive import ctx, TextGeneration, MultimodalContent, ResourceContent
from draive.openai import OpenAI, OpenAIResponsesConfig

async with ctx.scope(
    "image_analysis",
    OpenAIResponsesConfig(model="gpt-4o"),
    disposables=(OpenAI(),),
):
    # Create multimodal content with image
    content = MultimodalContent.of(
        "Analyze this image:",
        ResourceContent.of(image_bytes, mime_type="image/jpeg"),
    )

    # Generate analysis
    analysis = await TextGeneration.generate(
        instructions="Describe what you see in the image",
        input=content
    )


# Process document with mixed content
from draive import MultimodalTag

document = MultimodalContent.of(
    "Document Analysis Report",
    MultimodalTag.of(TextContent.of("Q1 Sales Report"), name="title"),
    "Summary: Sales increased by 15%",
    chart_image,
    "Detailed breakdown:",
    profile_artifact,
)

# Extract different content types
title_tag = document.tag("title")
texts = document.texts()
charts = document.images()

# Check content characteristics
if content.contains_resources:
    print("Content contains resources")

if content.contains_artifacts:
    print("Content contains artifacts")

if len(content.images()) > 0:

2. **Leverage metadata**: Use part-level metadata to add context and categorization
3. **Filter efficiently**: Use built-in filtering methods rather than manual iteration


```python
from collections.abc import Sequence
from draive import ctx, ModelGeneration, DataModel
from draive.openai import OpenAI, OpenAIResponsesConfig

class ImageDescription(DataModel):
    description: str
    objects: Sequence[str]

async with ctx.scope(
    "multimodal_generation",
    OpenAIResponsesConfig(model="gpt-4o"),
    disposables=(OpenAI(),),
):
    result = await ModelGeneration.generate(
        ImageDescription,
        instructions="Analyze the image and extract key information",
        input=multimodal_content
    )
```

The multimodal content system in Draive provides a powerful and flexible foundation for building applications that work with diverse content types while maintaining type safety and ease of use.
