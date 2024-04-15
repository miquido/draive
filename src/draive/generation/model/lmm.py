from collections.abc import Iterable

from draive.lmm import LMMCompletionMessage, lmm_completion
from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__: list[str] = [
    "lmm_generate_model",
]


async def lmm_generate_model[Generated: Model](
    model: type[Generated],
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, Generated]] | None = None,
) -> Generated:
    system_message: LMMCompletionMessage = LMMCompletionMessage(
        role="system",
        content=INSTRUCTION.format(
            instruction=instruction,
            format=model.specification(),
        ),
    )
    input_message: LMMCompletionMessage = LMMCompletionMessage(
        role="user",
        content=input,
    )

    context: list[LMMCompletionMessage]

    if examples:
        context = [
            system_message,
            *[
                message
                for example in examples
                for message in [
                    LMMCompletionMessage(
                        role="user",
                        content=example[0],
                    ),
                    LMMCompletionMessage(
                        role="assistant",
                        content=example[1].as_json(indent=2),
                    ),
                ]
            ],
            input_message,
        ]

    else:
        context = [
            system_message,
            input_message,
        ]

    completion: LMMCompletionMessage = await lmm_completion(
        context=context,
        tools=tools,
        output="json",
    )
    generated: Generated = model.from_json(completion.content_string)

    return generated


INSTRUCTION: str = """\
{instruction}

IMPORTANT!
The result have to conform to the following JSON Schema:
```
{format}
```
Provide ONLY a single, raw, valid JSON without any comments or formatting.
"""
