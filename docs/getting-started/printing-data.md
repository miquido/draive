<!-- markdownlint-disable-file MD046 -->
# Printing data

In **Draive** there are a few option for converting multimodal data into printable objects:

* `x.to_mapping()` *(Default)* - converts data into python `dict` object that is printable
* `x.to_json(indent: int | None = None)` - converts data into JSON object string
* `x.to_str()` - converts data into string using class method (based on class implementation)
* `str(x)` - uses default string method (`str(x.to_mapping())` for `DataModel` class)

Different Data Model objects in **Draive** will be displayed differently while using `s.to_str()` printing method.

The rest of this page will guide you through the most important data models and provide you with examples of theirs printing method.

## `TextContent`

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
Hello World!
```

## `ResourceContent`

Generates a Markdown media reference. Depending on `include_data`, the reference either embeds the base64 payload directly or uses a redacted placeholder.

!!! Note

    `kind` variable will be one of: `'image' | 'audio' | 'video' | ''`

* if `include_data` parameter in `to_str()` is `True`

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

* if `include_data` parameter in `to_str()` is `False`

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

## `ArtifactContent`

Produces delegated artifact output when visible, otherwise suppresses content
entirely for hidden artifacts.

* if `self.hidden` attribute is `True`

    ```python
    return ""
    ```

* if `self.hidden` attribute is `False`

    ```python
    return f"{self.artifact.to_str()}"
    ```

## `MultimodalTag`

Formats XML-like tags, optionally wrapping the rendered child content. Empty
content yields a self-closing tag.

* if `self.content` is empty

    ```python
    return f"<{self.name}{_tag_attributes(self.meta)}/>"
    ```

    **Example:**

    ```text
    <TAG_NAME attr_1 attr_2="val_2"/>
    ```

* if `self.content` is not empty

    ```python
    return f"<{self.name}{_tag_attributes(self.meta)}>{self.content.to_str()}</{self.name}>"
    ```

    **Example:**

    ```text
    <TAG_NAME attr_1 attr_2="val_2">Hello World!</TAG_NAME>
    ```

!!! Important

    `MultimodalTag` is the only Multimodal element that has a visible Meta section. Data from `meta` will end up as a XML tag attributes

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
