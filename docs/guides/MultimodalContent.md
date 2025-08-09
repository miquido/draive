# Multimodal Content

Draive provides comprehensive support for multimodal content through the `MultimodalContent` class and related types. This allows you to seamlessly work with text, images, audio, video, and other media types in your LLM applications.

## Core Components

The multimodal system consists of several key components:

### MultimodalContent

The main container for multimodal data that can hold various content elements:

```python
from draive import MultimodalContent, TextContent, MediaData

# Create multimodal content from various sources
content = MultimodalContent.of(
    "Here's an image:",
    MediaData.of(
        data=image_bytes,
        media="image/jpeg"
    ),
    "And some additional text."
)
```

### Content Elements

The framework supports these content element types:

- **TextContent**: Plain text with optional metadata
- **MediaContent**: Images, audio, video (data or references)
- **MetaContent**: Categorized content with metadata
- **DataModel**: Any structured data model

```python
from draive import TextContent, MediaData, MediaReference, MetaContent

# Text content
text = TextContent.of("Hello world!")

# Media data (embedded)
image_data = MediaData.of(
    image_bytes,
    media="image/png"
)

# Media reference (URL)
image_ref = MediaReference.of(
    "https://example.com/image.jpg",
    media="image/jpeg"
)

# Meta content with categorization
meta = MetaContent.of(
    "transcription",
    content=TextContent.of("Text...")
)
```

## Working with Multimodal Content

### Creating Content

```python
from draive import MultimodalContent

# From string
content = MultimodalContent.of("Simple text")

# From multiple elements
content = MultimodalContent.of(
    "Description:",
    image_data,
    "Additional context"
)

# With metadata
from haiway import Meta

content = MultimodalContent.of(
    "Content with metadata",
    meta=Meta.of({"source": "user", "timestamp": "2024-01-01"})
)
```

### Filtering and Accessing Content

```python
# Check for media presence
if content.has_media:
    print("Content contains media")

# Get all media
all_media = content.media()

# Get specific media types
images = content.media("image")
audio_files = content.media("audio")

# Filter content by type
text_only = content.text()
images_only = content.images()
audio_only = content.audio()
video_only = content.video()

# Remove media
text_content = content.without_media()
```

### Working with Artifacts

Artifacts are structured data models embedded in multimodal content. Any instance of DataModel which is not any of predefined content types is treated as artifact:

```python
from draive import DataModel

class UserProfile(DataModel):
    name: str
    age: int

# Content with artifacts
content = MultimodalContent.of(
    "User information:",
    UserProfile(name="John", age=30)
)

# Check for artifacts
if content.has_artifacts:
    # Get all artifacts
    all_artifacts = content.artifacts()
    
    # Get specific artifact type
    profiles = content.artifacts(UserProfile)

# Remove artifacts
clean_content = content.without_artifacts()
```

When an artifact is presented to LLM it is converted to its json representation automatically.

### Meta Content Operations

```python
# Get meta content
meta_items = content.meta()

# Get by category
instructions = content.meta(category="instruction")

# Remove meta content
clean_content = content.without_meta()
```

## Content Manipulation

### Appending Content

```python
# Append new content
extended = content.appending(
    "Additional text",
    new_image_data
)

# Extend with other multimodal content
other_content = MultimodalContent.of("More content")
combined = content.extended_by(other_content)
```

### String Conversion

```python
# Convert to string
text_repr = content.to_str()

# With custom joiner between parts
text_repr = content.to_str(joiner=" | ")

# Include media data URIs
text_with_data = content.to_str(include_data=True)

# Direct string conversion
text_repr = str(content)
```

## Practical Examples

### Image Analysis

```python
from draive import ctx, TextGeneration, MultimodalContent, MediaData
from draive.openai import OpenAI, OpenAIResponsesConfig

async with ctx.scope(
    "image_analysis",
    OpenAIResponsesConfig(model="gpt-4o"),
    disposables=(OpenAI(),),
):
    # Create multimodal content with image
    content = MultimodalContent.of(
        "Analyze this image:",
        MediaData.of(
            data=image_bytes,
            media="image/jpeg"
        )
    )
    
    # Generate analysis
    analysis = await TextGeneration.generate(
        instructions="Describe what you see in the image",
        input=content
    )
    
    print(analysis)
```

### Document Processing

```python
# Process document with mixed content
document = MultimodalContent.of(
    "Document Analysis Report",
    MetaContent.of("title", content=TextContent.of("Q1 Sales Report")),
    "Summary: Sales increased by 15%",
    chart_image,
    "Detailed breakdown:",
    table_data_model
)

# Extract different content types
title = document.meta(category="title")
text_content = document.text()
charts = document.images()
data_models = document.artifacts()
```

### Content Validation

```python
# Check content types
if content.is_media("image"):
    print("Content is image-only")

if content.is_artifact():
    print("Content contains only artifacts")

if content.is_meta():
    print("Content is meta-only")

# Boolean check for non-empty content
if content:
    print("Content is not empty")
```

## Best Practices

1. **Use appropriate media types**: Always specify correct MIME types for media content
2. **Leverage metadata**: Use Meta objects to add context and categorization
3. **Filter efficiently**: Use built-in filtering methods rather than manual iteration
4. **Handle empty content**: Always check if content exists before processing
5. **Optimize media handling**: Use references for large media files when possible

## Integration with LLM Providers

MultimodalContent seamlessly integrates with various LLM providers that support multimodal input:

```python
from collections.abc import Sequence
from draive import ModelGeneration
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
