<!-- markdownlint-disable-file MD046 -->
# Printing data

In **Draive** you often need to surface multimodal data in logs, dashboards, or CLI output. Every model state exposes a few helpers that flatten structured content into printable strings:

| Helper | Output |
| --- | --- |
| `x.to_mapping()` *(default)* | `dict` representation |
| `x.to_json(indent: int \| None = None)` | JSON string |
| `x.to_str()` | Model-defined string |
| `str(x)` | Alias for `str(x.to_mapping())` on `DataModel` |

Different `DataModel` instances implement `.to_str()` in their own way, so the rendered view varies by content type. The sections below walk through the most common multimodal classes and show how their printing helpers behave.

## TextContent

Plain text output that mirrors the original `text` value without any extra
markup or quoting.

```python
return f"{self.text}"
```

**Example:**

```python
text: TextContent = TextContent.of('Hello world!')
ctx.log_info(text.to_str())
```

```text
Hello world!
```

## ResourceContent

Generates a Markdown media reference. Depending on `include_data`, the reference either embeds the base64 payload directly or uses a redacted placeholder.

!!! note

    `kind` variable will be one of: `'image' | 'audio' | 'video' | ''`

When `include_data=True`, `ResourceContent.to_str()` returns the full base64 payload:

```python
return f"![{kind}](data:{self.mime_type};base64,{self.data})"
```

**Example:**

```python
resource: ResourceContent = ResourceContent.of(b'FF', mime_type='application/octet-stream')
ctx.log_info(resource.to_str(include_data=True))
```

```text
![](data:application/octet-stream;base64,RkY=)
```

With the default `include_data=False`, it emits a placeholder that keeps the media type visible without leaking the bytes:

```python
return f"![{kind}](REDACTED)"
```

**Example:**

```python
with open('./src/dog.jpg', 'rb') as img:
    img_data: bytes = img.read()
resource: ResourceContent = ResourceContent.of(img_data, mime_type='image/jpeg')
ctx.log_info(resource.to_str())
```

```text
![image](REDACTED)
```

## ArtifactContent

Produces delegated artifact output when visible, otherwise suppresses content
entirely for hidden artifacts.

Hidden artifacts render as an empty string:

```python
return ""
```

Visible artifacts reuse the nested artifact’s `to_str()` result:

```python
return f"{self.artifact.to_str()}"
```

## MultimodalTag

Formats XML-like tags, optionally wrapping the rendered child content. Empty
content yields a self-closing tag.

When `self.content` is empty, the printer emits a self-closing tag:

```python
return f"<{self.name}{_tag_attributes(self.meta)}/>"
```

**Example:**

```text
<TAG_NAME attr_1 attr_2="val_2"/>
```

Otherwise it wraps the rendered child content:

```python
return f"<{self.name}{_tag_attributes(self.meta)}>{self.content.to_str()}</{self.name}>"
```

**Example:**

```text
<TAG_NAME attr_1 attr_2="val_2">Hello World!</TAG_NAME>
```

!!! important

    `MultimodalTag` is the only multimodal element that exposes metadata inline. Values stored in `meta` appear as XML-style tag attributes.

## MultimodalContent

Concatenates the string form of each part, resulting in a single composite
response without delimiters.

```python
return "".join(part.to_str() for part in self.parts)
```

**Example:**

```python
from draive import MultimodalContent, TextContent, ctx

multimodal: MultimodalContent = MultimodalContent.of(
    TextContent.of('Hello '),
    TextContent.of('World!'),
)
ctx.log_info(multimodal.to_str())
```

Result:

```text
Hello World!
```
