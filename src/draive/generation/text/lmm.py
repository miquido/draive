from collections.abc import Iterable
from typing import Any

from draive.lmm import LMMMessage, lmm_completion
from draive.tools import Toolbox
from draive.types import MultimodalContent

__all__: list[str] = [
    "lmm_generate_text",
]


async def lmm_generate_text(
    *,
    instruction: str,
    input: MultimodalContent,  # noqa: A002
    tools: Toolbox | None = None,
    examples: Iterable[tuple[MultimodalContent, str]] | None = None,
    **extra: Any,
) -> str:
    system_message: LMMMessage = LMMMessage(
        role="system",
        content=instruction,
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
                        content=example[1],
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
        output="text",
        stream=False,
        **extra,
    )
    generated: str = completion.content_string

    return generated
