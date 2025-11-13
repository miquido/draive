<!-- markdownlint-disable-file MD046 -->

# Multimodal data

One of the most important skills when using **Draive** is fully understanding how to treat
multimodal data and model context.

Any data can be converted into the `Multimodal` alias (text, audio, image, or artifact; any
`DataModel`). This multimodal representation is usually the input and output type for most
**Draive** functions.

Multimodal data creates the model context (user inputs and model outputs), so understanding it is
crucial.

## Multimodal

The `Multimodal` alias merges `MultimodalContent`, `MultimodalTag`, and others. In some places input
parameters are typed as `Multimodal` for automated transformation.

![Multimodal Class](../diagrams/out/Multimodal.svg)

!!! Note

    Constructors and helpers such as `MultimodalContent.of(*elements: "Multimodal")` use the `Multimodal` alias to normalize any mix of multimodal parts into one consistent `MultimodalContent`.

## Multimodal Content

`MultimodalContent` is the central container bundling the different `MultimodalContentPart` blocks.
Concrete parts carry concrete payloads: plain text (`TextContent`), references or embedded resources
(`ResourceReference`, `ResourceContent`), artifacts (`ArtifactContent`), and tags (`MultimodalTag`).
As a result any span of content can be examined as a set of selected parts or as an entire tag tree.

![MultimodalContent aggregates](../diagrams/out/MultimodalContent.svg)

`MultimodalContent` serves as the input or output in multiple places in the **Draive** framework. It
can be used for filtering, splitting, replacing, and other operations.

**A few examples of MultimodalContent usage in Draive**:

- `@tool` decorator decorates functions that return `MultimodalContent` (see
  [Basic tools use](../guides/BasicToolsUse.md))
- `ModelInput` and `ModelOutput` classes use `MultimodalContent`
- `TextGeneration.generate(...)` accepts `MultimodalContent` as input (see
  [Basic usage](../guides/BasicUsage.md))

`MultimodalContent` is an **easy-to-use and intelligent** class that does a lot for you under the
hood:

1. It avoids extra nesting:

    ```python
    inner_multimodal = MultimodalContent.of("Hello world!")
    print(inner_multimodal)  # {'type': 'content', 'parts': [{'text': 'Hello world!', 'meta': {}}]}
    outer_multimodal = MultimodalContent.of(inner_multimodal)
    print(outer_multimodal)  # {'type': 'content', 'parts': [{'text': 'Hello world!', 'meta': {}}]}
    # (Same as the first one despite nesting)
    ```

1. It merges multiple parts if the types match:

    ```python
    class User(DataModel):
        first_name: str
        last_name: str

    content = MultimodalContent.of(
        MultimodalTag.of(
            MultimodalTag.of(
                "Hello",
                name="inner",
            ),
            ArtifactContent.of(
                User(
                    first_name="James",
                    last_name="Smith",
                )
            ),
            name="outer",
        )
    )

    print(content)
    # {
    #    'type': 'content', 
    #    'parts': [
    #        {
    #            'text': '<outer><inner>Hello</inner>', 
    #            'meta': {}
    #        }, 
    #        {
    #            'category': 'User', 
    #            'artifact': {
    #                'first_name': 'James', 
    #                'last_name': 'Smith'
    #            }, 
    #            'hidden': False, 
    #            'meta': {}
    #        }, 
    #        {
    #            'text': '</outer>', 
    #            'meta': {}
    #        }
    #    ]
    # }
    ```

    !!! Note

        `MultimodalTag` produces text parts, so they merge with `TextContent`.

1. Comes with a set of helper functions to speed up your work. Examples:

    ```python
    print(multimodal.texts())
    # (
    #    {'text': '<outer><inner>Hello</inner>', 'meta': {}},
    #    {'text': '</outer>', 'meta': {}}
    # )
    print(multimodal.tags())
    # (
    #    {'name': 'outer', 'content': {...}, 'meta': {}},
    #    {'name': 'inner', 'content': {...}, 'meta': {}}
    # )
    print(multimodal.artifacts())
    # (
    #    {
    #        'category': 'User',
    #        'artifact': {
    #            'first_name': 'James',
    #            'last_name': 'Smith'
    #        },
    #        'hidden': False,
    #        'meta': {}
    #    },
    # )
    ```

    !!! Tip

        `MultimodalContent` has more ready-to-use methods for filtering such as `matching_meta()`, `split_by_meta()`, `without_resources()` or `audio()`. This is another argument to use `MultimodalContent` rather than other data models

## Model Input

`ModelInput` groups the user-provided blocks (`ModelInputBlock`). Each block is backed by
`MultimodalContent`, so it preserves every part type described above. The input stream may also
embed tool responses (`ModelToolResponse`), which return their payload as `MultimodalContent`,
letting you treat tool output the same way as regular text-and-media blocks.

![ModelInput class relationships](../diagrams/out/ModelInput.svg){style="height:400px; margin: auto; display: block;"}

!!! note

    Note that `ModelToolResponse` has a `content` attribute of type `MultimodalContent`. This is the reason why `@tool` decorated functions must return `MultimodalContent`.

## Model Output

`ModelOutput` mirrors the same structure on the response side. Output blocks (`ModelOutputBlock`)
expose `MultimodalContent`, and both the model's reasoning trail (`ModelReasoning`) and its tool
requests (`ModelToolRequest`) rely on the same container. This means the full generation flow - from
visible content to internal thinking and tool invocations - can be analysed with one coherent set of
helpers.

![ModelOutput class relationships](../diagrams/out/ModelOutput.svg){style="height:400px; margin: auto; display: block;"}

!!! tip

    There are ready-to-use methods like `without_tools()` to get model output without blocks related to tool requests and responses, or `reasoning()` to get model reasoning blocks. That can help you implement your features easily.
