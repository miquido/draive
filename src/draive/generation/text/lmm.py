from collections.abc import Iterable

from draive.lmm import LMMCompletionMessage, lmm_completion
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
) -> str:
    system_message: LMMCompletionMessage = LMMCompletionMessage(
        role="system",
        content=instruction,
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

    completion: LMMCompletionMessage = await lmm_completion(
        context=context,
        tools=tools,
        output="text",
    )
    generated: str = completion.content_string

    return generated
