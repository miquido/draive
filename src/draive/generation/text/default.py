from collections.abc import Iterable
from typing import Any

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox, lmm_invocation
from draive.types import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponse,
    Multimodal,
    MultimodalContent,
)

__all__: list[str] = [
    "default_generate_text",
]


async def default_generate_text(
    *,
    instruction: Instruction | str | None,
    input: Multimodal,  # noqa: A002
    prefill: str | None,
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
            LMMInput.of(input),
        ]

        recursion_level: int = 0
        while recursion_level <= toolbox.recursion_limit:
            match await lmm_invocation(
                instruction=instruction,
                context=[*context, LMMCompletion.of(prefill)] if prefill else context,
                tools=toolbox.available_tools(),
                tool_selection=toolbox.tool_selection(recursion_level=recursion_level),
                output="text",
                **extra,
            ):
                case LMMCompletion() as completion:
                    ctx.log_debug("Received text generation result")
                    return completion.content.as_string()

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug("Received text generation tool calls")
                    responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

                    if direct_responses := [response for response in responses if response.direct]:
                        return MultimodalContent.of(
                            *[response.content for response in direct_responses]
                        ).as_string()

                    else:
                        context.extend([tool_requests, *responses])

            recursion_level += 1  # continue with next recursion level

        raise RuntimeError("LMM exceeded limit of recursive calls")
