from collections.abc import Iterable
from typing import Any

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInstruction,
    LMMToolRequests,
    LMMToolResponses,
)
from draive.multimodal import MultimodalContent
from draive.prompts import Prompt
from draive.tools import Toolbox

__all__ = ("generate_text",)


async def generate_text(
    *,
    instruction: Instruction | None,
    input: Prompt | MultimodalContent,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[MultimodalContent, str]],
    **extra: Any,
) -> str:
    async with ctx.scope("generate_text"):
        context: list[LMMContextElement] = [
            *[
                message
                for example in examples
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

        return await _text_generation(
            instruction=instruction,
            context=context,
            toolbox=toolbox,
        )


async def _text_generation(
    instruction: Instruction | None,
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> str:
    formatted_instruction: LMMInstruction | None = Instruction.formatted(instruction)
    tools_turn: int = 0
    result: MultimodalContent = MultimodalContent.empty
    result_extension: MultimodalContent = MultimodalContent.empty
    while True:
        ctx.log_debug("...requesting completion...")
        match await LMM.completion(
            instruction=formatted_instruction,
            context=context,
            tools=toolbox.available_tools(tools_turn=tools_turn),
            output="text",
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("...received result...")
                result = result_extension.appending(completion.content)
                break  # proceed to resolving

            case LMMToolRequests() as tool_requests:
                ctx.log_debug(f"...received tool requests (turn {tools_turn})...")
                # skip tool_requests.content - no need for extra comments
                tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                if completion := tool_responses.completion(extension=result_extension):
                    ctx.log_debug("...received tools direct result...")
                    result = completion.content
                    break  # proceed to resolving

                elif extension := tool_responses.completion_extension():
                    ctx.log_debug("...received tools result extension...")
                    result_extension = result_extension.appending(extension)

                ctx.log_debug("...received tools responses...")
                context.extend((tool_requests, tool_responses))

        tools_turn += 1  # continue with next turn

    ctx.log_debug("...text completed!")
    return result.to_str()
