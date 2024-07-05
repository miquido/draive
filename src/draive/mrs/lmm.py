import json
from collections.abc import AsyncGenerator, AsyncIterable, Sequence
from typing import Any, Literal, overload

from mistralrs import (  # type: ignore
    ChatCompletionChunkResponse,
    ChatCompletionResponse,
    ResponseMessage,
)

from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.mrs.client import MRSClient
from draive.mrs.config import MRSChatConfig
from draive.mrs.errors import MRSException
from draive.parameters import ToolSpecification
from draive.scope import ctx
from draive.types import (
    Instruction,
    LMMCompletion,
    LMMCompletionChunk,
    LMMContextElement,
    LMMInput,
    LMMOutput,
    LMMOutputStream,
    LMMOutputStreamChunk,
    LMMToolRequests,
    LMMToolResponse,
)

__all__ = [
    "mrs_lmm_invocation",
]


@overload
async def mrs_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def mrs_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMOutput: ...


@overload
async def mrs_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def mrs_lmm_invocation(  # noqa: PLR0913
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput:
    with ctx.nested(
        "mrs_lmm_completion",
        metrics=[
            ArgumentsTrace.of(
                instruction=instruction,
                context=context,
                tools=tools,
                tool_requirement=tool_requirement,
                output=output,
                stream=stream,
                **extra,
            ),
        ],
    ):
        ctx.log_debug("Requested mistral.rs lmm")
        client: MRSClient = ctx.dependency(MRSClient)
        config: MRSChatConfig = ctx.state(MRSChatConfig).updated(**extra)
        ctx.record(config)

        if tools:
            ctx.log_warning(
                "Attempting to use mistral.rs with tools which is not supported yet."
                " Ignoring provided tools..."
            )

        match output:
            case "text":
                config = config.updated(response_format={"type": "text"})

            case "json":
                config = config.updated(response_format={"type": "json_object"})

        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": Instruction.of(instruction).format(),
            },
            *[_convert_context_element(element=element) for element in context],
        ]

        if stream:
            return ctx.stream(
                generator=_chat_completion_stream(
                    client=client,
                    config=config,
                    messages=messages,
                ),
            )

        else:
            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
            )


def _convert_context_element(
    element: LMMContextElement,
) -> dict[str, object]:
    match element:
        case LMMInput() as input:
            return {
                "role": "user",  # multimodal not supported yet
                "content": input.content.as_string(),
            }

        case LMMCompletion() as completion:
            return {
                "role": "assistant",
                "content": completion.content.as_string(),
            }

        # tools are not supported yet
        case LMMToolRequests() as tool_requests:
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": request.identifier,
                        "type": "function",
                        "function": {
                            "name": request.tool,
                            "arguments": json.dumps(request.arguments),
                        },
                    }
                    for request in tool_requests.requests
                ],
            }

        # tools are not supported yet
        case LMMToolResponse() as tool_response:
            return {
                "role": "tool",
                "tool_call_id": tool_response.identifier,
                "content": tool_response.content.as_string(),
            }


async def _chat_completion(
    *,
    client: MRSClient,
    config: MRSChatConfig,
    messages: list[dict[str, object]],
) -> LMMOutput:
    completion: ChatCompletionResponse = await client.chat_completion(
        config=config,
        messages=messages,
    )

    if usage := completion.usage:
        ctx.record(
            TokenUsage.for_model(
                config.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            ),
        )

    if not completion.choices:
        raise MRSException("Invalid mistral.rs completion - missing messages!", completion)

    completion_message: ResponseMessage = completion.choices[0].message

    if message := completion_message.content:
        ctx.record(ResultTrace.of(message))
        return LMMCompletion.of(message)

    else:
        raise MRSException("Invalid mistral.rs completion", completion)


async def _chat_completion_stream(
    *,
    client: MRSClient,
    config: MRSChatConfig,
    messages: list[dict[str, object]],
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    completion_stream: AsyncIterable[ChatCompletionChunkResponse] = await client.chat_completion(
        config=config,
        messages=messages,
        stream=True,
    )

    accumulated_completion: str = ""
    async for part in completion_stream:
        # we always request only one
        part_text: str = part.choices[0].delta.content
        accumulated_completion += part_text
        yield LMMCompletionChunk.of(part_text)

    ctx.record(ResultTrace.of(accumulated_completion))
