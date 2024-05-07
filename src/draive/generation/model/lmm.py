from collections.abc import Iterable
from typing import Any

from draive.lmm import LMMMessage, lmm_completion
from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__: list[str] = [
    "lmm_generate_model",
]


async def lmm_generate_model[Generated: Model](
    generated: type[Generated],
    /,
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, Generated]] | None = None,
    **extra: Any,
) -> Generated:
    system_message: LMMMessage = LMMMessage(
        role="system",
        content=INSTRUCTION.format(
            instruction=instruction,
            format=generated.specification(),
        ),
    )
    input_message: LMMMessage = LMMMessage(
        role="user",
        content=input,
    )

    context: list[LMMMessage]

    if examples:
        context = [
            system_message,
            *[
                message
                for example in examples
                for message in [
                    LMMMessage(
                        role="user",
                        content=example[0],
                    ),
                    LMMMessage(
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

    completion: LMMMessage = await lmm_completion(
        context=context,
        tools=tools,
        output="json",
        stream=False,
        **extra,
    )

    return generated.from_json(completion.content_string)


INSTRUCTION: str = """\
{instruction}

IMPORTANT!
The result have to conform to the following JSON Schema:
```
{format}
```
Provide ONLY a single, raw, valid JSON without any comments or formatting.
"""
