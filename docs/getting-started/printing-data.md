<!-- markdownlint-disable-file MD046 -->

# Printing data

In **Draive** you often need to surface multimodal data in logs, dashboards, or CLI output. Every
model state exposes a few helpers that flatten structured content into printable strings:

| Helper                                  | Output                                             |
| --------------------------------------- | -------------------------------------------------- |
| `x.to_mapping(recursive: bool = True)`  | `Mapping` representation                           |
| `x.to_json(indent: int \| None = None)` | JSON string                                        |
| `x.to_str()`                            | Model-defined string, defaults to `str(x)`         |
| `str(x)`                                | `ClassName(field: value)` for typed `State` models |

Different model instances implement `.to_str()` in their own way, so the rendered view varies
by content type. The sections below walk through the most common multimodal classes and show how
their printing helpers behave.

## TextContent

Plain text output that mirrors the original `text` value without any extra markup or quoting.

```python
return self.text
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

Generates a Markdown media reference. Depending on `include_data`, the reference either embeds the
base64 data URI directly or leaves the link empty.

!!! Note

    `kind` variable will be one of: `'image' | 'audio' | 'video' | ''`

When `include_data=True`, `ResourceContent.to_str()` returns the full base64 payload:

```python
return f"![{kind}]({self.to_data_uri()})"
```

**Example:**

```python
resource: ResourceContent = ResourceContent.of(b'FF', mime_type='application/octet-stream')
ctx.log_info(resource.to_str(include_data=True))
```

```text
![](data:application/octet-stream;base64,RkY=)
```

With the default `include_data=False`, it keeps the media kind visible without leaking the bytes:

```python
return f"![{kind}]()"
```

**Example:**

```python
with open('./src/dog.jpg', 'rb') as img:
    img_data: bytes = img.read()
resource: ResourceContent = ResourceContent.of(img_data, mime_type='image/jpeg')
ctx.log_info(resource.to_str())
```

```text
![image]()
```

## ArtifactContent

Renders the wrapped artifact payload as indented JSON when visible, otherwise suppresses content
entirely for hidden artifacts.

Hidden artifacts render as an empty string:

```python
return ""
```

Visible artifacts are serialized to JSON:

```python
return json.dumps(self.artifact, indent=2)
```

**Example:**

```python
from draive import ArtifactContent, State, ctx


class User(State, serializable=True):
    first_name: str


artifact: ArtifactContent = ArtifactContent.of(User(first_name="James"), category="profile")
ctx.log_info(artifact.to_str())
```

```text
{
  "first_name": "James"
}
```

## MultimodalTag

Formats XML-like tags, optionally wrapping the rendered child content. Empty content yields a
self-closing tag.

When `self.content` is empty, the printer emits a self-closing tag:

```python
return f"<{self.name}{_tag_attributes(self.meta)}/>"
```

**Example:**

```text
<TAG_NAME attr_1="True" attr_2="val_2"/>
```

Otherwise it wraps the rendered child content:

```python
return f"<{self.name}{_tag_attributes(self.meta)}>{self.content.to_str()}</{self.name}>"
```

**Example:**

```text
<TAG_NAME attr_1="True" attr_2="val_2">Hello World!</TAG_NAME>
```

!!! Important

    `MultimodalTag` is the only multimodal element that exposes metadata inline. Values stored in `meta` appear as quoted XML-style tag attributes.

## MultimodalContent

Concatenates the string form of each part, resulting in a single composite response without
delimiters. An optional `artifact_to_str` argument overrides the rendering of `ArtifactContent`
parts.

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
