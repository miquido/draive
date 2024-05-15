from collections.abc import Iterable
from typing import Any

from draive.lmm import LMMMessage, lmm_completion
from draive.tools import Toolbox
from draive.types import Model, MultimodalContent

__all__: list[str] = [
    "lmm_generate_model",
]


async def lmm_generate_model[Generated: Model](  # noqa: PLR0913
    generated: type[Generated],
    /,
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    schema_variable: str | None = None,
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, Generated]] | None = None,
    **extra: Any,
) -> Generated:
    system_message: LMMMessage
    if variable := schema_variable:
        system_message = LMMMessage(
            role="system",
            content=instruction.format(**{variable: generated.specification()}),
        )

    else:
        system_message = LMMMessage(
            role="system",
            content=DEFAULT_INSTRUCTION.format(
                instruction=instruction,
                schema=generated.specification(),
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


DEFAULT_INSTRUCTION: str = """\
{instruction}

IMPORTANT!
The result have to conform to the following JSON Schema:
```
{schema}
```
Provide ONLY a single, raw, valid JSON without any comments or formatting.
"""
