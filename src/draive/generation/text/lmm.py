from collections.abc import Iterable, Sequence
from typing import Any

from draive.lmm import AnyTool, Toolbox, lmm_invocation
from draive.scope import ctx
from draive.types import (
    Instruction,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInstruction,
    LMMToolRequests,
    LMMToolResponse,
    MultimodalContent,
    MultimodalContentElement,
)

__all__: list[str] = [
    "lmm_generate_text",
]


async def lmm_generate_text(
    *,
    instruction: Instruction | str,
    input: MultimodalContent | MultimodalContentElement,  # noqa: A002
    tools: Toolbox | Sequence[AnyTool] | None = None,
    examples: Iterable[tuple[MultimodalContent | MultimodalContentElement, str]] | None = None,
    **extra: Any,
) -> str:
    with ctx.nested("lmm_generate_text"):
        toolbox: Toolbox
        match tools:
            case None:
                toolbox = Toolbox()

            case Toolbox() as tools:
                toolbox = tools

            case [*tools]:
                toolbox = Toolbox(*tools)

        context: list[LMMContextElement] = [
            LMMInstruction.of(instruction),
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

        for recursion_level in toolbox.call_range:
            match await lmm_invocation(
                context=context,
                tools=toolbox.available_tools(recursion_level=recursion_level),
                require_tool=toolbox.tool_suggestion(recursion_level=recursion_level),
                output="text",
                stream=False,
                **extra,
            ):
                case LMMCompletion() as completion:
                    ctx.log_debug("Received text generation result")
                    return completion.content.as_string()

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug("Received text generation tool calls")
                    context.append(tool_requests)
                    responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

                    if direct_responses := [response for response in responses if response.direct]:
                        return MultimodalContent.of(
                            *[response.content for response in direct_responses]
                        ).as_string()

                    else:
                        context.extend(responses)

    # fail if we have not provided a result until this point
    raise RuntimeError("Failed to produce conversation completion")
