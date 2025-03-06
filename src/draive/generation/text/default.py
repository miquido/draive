from collections.abc import Iterable
from typing import Any

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponses,
    lmm_invoke,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import AnyTool, Toolbox

__all__ = [
    "default_generate_text",
]


async def default_generate_text(
    *,
    instruction: Instruction | str | None,
    input: Prompt | Multimodal,  # noqa: A002
    tools: Toolbox | Iterable[AnyTool] | None,
    examples: Iterable[tuple[Multimodal, str]] | None,
    **extra: Any,
) -> str:
    with ctx.scope("generate_text"):
        toolbox: Toolbox = Toolbox.of(tools)

        context: list[LMMContextElement] = [
            *[
                message
                for example in examples or []
                for message in [
                    LMMInput.of(example[0]),
                    LMMCompletion.of(example[1]),
                ]
            ],
        ]

        match input:
            case Prompt() as prompt:
                context.extend(prompt.content)

            case value:
                context.append(LMMInput.of(value))

        recursion_level: int = 0
        while recursion_level <= toolbox.repeated_calls_limit:
            match await lmm_invoke(
                instruction=instruction,
                context=context,
                tools=toolbox.available_tools(),
                tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
                output="text",
                **extra,
            ):
                case LMMCompletion() as completion:
                    ctx.log_debug("Received text generation result")
                    return completion.content.as_string()

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug("Received text generation tool calls")
                    tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                    if direct_responses := [
                        response for response in tool_responses.responses if response.direct
                    ]:
                        return MultimodalContent.of(
                            *[response.content for response in direct_responses]
                        ).as_string()

                    else:
                        context.extend(
                            [
                                tool_requests,
                                tool_responses,
                            ]
                        )

            recursion_level += 1  # continue with next recursion level

        raise RuntimeError("LMM exceeded limit of recursive calls")
