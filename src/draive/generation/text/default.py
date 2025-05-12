from asyncio import Task, gather
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import Any

from haiway import AsyncQueue, ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
)
from draive.lmm.types import LMMStreamChunk
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import Tool, Toolbox

__all__ = ("generate_text",)


async def generate_text(
    *,
    instruction: Instruction | str | None,
    input: Prompt | Multimodal,  # noqa: A002
    tools: Toolbox | Iterable[Tool] | None,
    examples: Iterable[tuple[Multimodal, str]] | None,
    stream: bool,
    **extra: Any,
) -> AsyncIterable[str] | str:
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

        if stream:
            return await _text_generation_stream(
                instruction=instruction,
                context=context,
                toolbox=toolbox,
            )

        else:
            return await _text_generation(
                instruction=instruction,
                context=context,
                toolbox=toolbox,
            )


async def _text_generation(
    instruction: Instruction | str | None,
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> str:
    recursion_level: int = 0
    while recursion_level <= toolbox.repeated_calls_limit:
        match await LMM.completion(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            output="text",
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("Received text generation result")
                return completion.content.to_str()

            case LMMToolRequests() as tool_requests:
                ctx.log_debug("Received text generation tool calls")
                tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                if direct_responses := [
                    response
                    for response in tool_responses.responses
                    if response.handling == "direct_result"
                ]:
                    return MultimodalContent.of(
                        *[response.content for response in direct_responses]
                    ).to_str()

                else:
                    context.extend(
                        [
                            tool_requests,
                            tool_responses,
                        ]
                    )

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")


async def _text_generation_stream(
    instruction: Instruction | str | None,
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncIterator[str]:
    output_queue = AsyncQueue[str]()

    async def consume_lmm_output() -> None:
        recursion_level: int = 0
        try:
            while recursion_level <= toolbox.repeated_calls_limit:
                pending_tool_requests: list[LMMToolRequest] = []
                pending_tool_responses: list[Task[LMMToolResponse]] = []
                accumulated_text: str = ""

                async for element in await LMM.completion(
                    instruction=instruction,
                    context=context,
                    tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
                    tools=toolbox.available_tools(),
                    output="text",
                    stream=True,
                    **extra,
                ):
                    match element:
                        case LMMStreamChunk() as chunk:
                            chunk_text: str = chunk.content.to_str()
                            accumulated_text += chunk_text
                            output_queue.enqueue(chunk_text)

                            if chunk.eod:
                                # end of streaming for text generation
                                return output_queue.finish()

                        case LMMToolRequest() as tool_request:
                            # we could start processing immediately
                            pending_tool_requests.append(tool_request)
                            pending_tool_responses.append(ctx.spawn(toolbox.respond, tool_request))

                assert pending_tool_requests  # nosec: B101
                tool_requests: LMMToolRequests = LMMToolRequests.of(
                    pending_tool_requests,
                    content=MultimodalContent.of(accumulated_text) if accumulated_text else None,
                )
                tool_responses: LMMToolResponses = LMMToolResponses.of(
                    await gather(*pending_tool_responses),
                )
                if direct_content := [
                    response.content
                    for response in tool_responses.responses
                    if response.handling == "direct_result"
                ]:
                    response_text: str = MultimodalContent.of(*direct_content).to_str()
                    output_queue.enqueue(response_text)
                    return output_queue.finish()

                else:
                    context.extend(
                        [
                            tool_requests,
                            tool_responses,
                        ]
                    )

                recursion_level += 1  # continue with next recursion level

            raise RuntimeError("LMM exceeded limit of tool calls")

        except BaseException as exc:
            output_queue.finish(exception=exc)
            raise exc

    ctx.spawn(consume_lmm_output)

    return output_queue
