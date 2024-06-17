import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, overload

from mistralrs import ChatCompletionResponse, ResponseMessage

from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.mrs.client import MRSClient
from draive.mrs.config import MRSChatConfig
from draive.mrs.errors import MRSException
from draive.parameters import ToolSpecification
from draive.scope import ctx
from draive.types import (
    LMMCompletion,
    LMMCompletionChunk,
    LMMContextElement,
    LMMInput,
    LMMInstruction,
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
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def mrs_lmm_invocation(
    *,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMOutput: ...


@overload
async def mrs_lmm_invocation(
    *,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def mrs_lmm_invocation(
    *,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput:
    with ctx.nested(
        "mrs_lmm_completion",
        metrics=[
            ArgumentsTrace.of(
                context=context,
                tools=tools,
                require_tool=require_tool,
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
            _convert_context_element(element=element) for element in context
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
        case LMMInstruction() as instruction:
            return {
                "role": "system",
                "content": instruction.content,
            }

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
    ctx.log_warning("mistral.rs streaming api is not supported yet, using regular response...")
    output: LMMOutput = await _chat_completion(
        client=client,
        config=config,
        messages=messages,
    )

    match output:
        case LMMCompletion() as completion:
            yield LMMCompletionChunk.of(completion.content)

        case other:
            yield other
