import json
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInvocation,
    LMMOutput,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponses,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.metrics import TokenUsage
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.mistral.models import ChatCompletionResponse, ChatMessage, ChatMessageResponse
from draive.mistral.types import MistralException
from draive.multimodal import MultimodalContent

__all__ = [
    "mistral_lmm",
]


def mistral_lmm(
    client: MistralClient | None = None,
    /,
) -> LMMInvocation:
    client = client or MistralClient.shared()

    async def lmm_invocation(  # noqa: PLR0913
        *,
        instruction: Instruction | str | None,
        context: Sequence[LMMContextElement],
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        prefill: MultimodalContent | None = None,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("mistral_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                )
            )
            config: MistralChatConfig = ctx.state(MistralChatConfig).updated(**extra)
            ctx.record(config)

            response_format: dict[str, str]
            match output:
                case "auto" | "text":
                    response_format = {"type": "text"}

                case "image":
                    raise NotImplementedError("image output is not supported by mistral")

                case "audio":
                    raise NotImplementedError("audio output is not supported by mistral")

                case "video":
                    raise NotImplementedError("video output is not supported by mistral")

                case _:
                    if tools:
                        ctx.log_warning(
                            "Attempting to use Mistral in JSON mode with tools which is not"
                            " supported. Using text mode instead..."
                        )
                        response_format = {"type": "text"}

                    else:
                        response_format = {"type": "json_object"}

            if prefill:
                context = [*context, LMMCompletion.of(prefill)]

            messages: list[ChatMessage] = list(
                chain.from_iterable(
                    [_convert_context_element(element=element) for element in context]
                )
            )

            if instruction:
                messages = [
                    ChatMessage(
                        role="system",
                        content=Instruction.formatted(instruction),
                    ),
                    *messages,
                ]

            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_selection=tool_selection,
            )

    return LMMInvocation(invoke=lmm_invocation)


def _convert_context_element(
    element: LMMContextElement,
) -> Iterable[ChatMessage]:
    match element:
        case LMMInput() as input:
            return (
                ChatMessage(
                    role="user",
                    content=input.content.as_string(),
                ),
            )

        case LMMCompletion() as completion:
            return (
                ChatMessage(
                    role="assistant",
                    content=completion.content.as_string(),
                ),
            )

        case LMMToolRequests() as tool_requests:
            return (
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": request.identifier,
                            "function": {
                                "name": request.tool,
                                "arguments": json.dumps(dict(request.arguments)),
                            },
                        }
                        for request in tool_requests.requests
                    ],
                ),
            )

        case LMMToolResponses() as tool_responses:
            return (
                ChatMessage(
                    role="tool",
                    tool_call_id=response.identifier,
                    name=response.tool,
                    content=response.content.as_string(),
                )
                for response in tool_responses.responses
            )


async def _chat_completion(  # noqa: PLR0913
    *,
    client: MistralClient,
    config: MistralChatConfig,
    messages: list[ChatMessage],
    response_format: dict[str, str],
    tools: Iterable[LMMToolSpecification] | None,
    tool_selection: LMMToolSelection,
) -> LMMOutput:
    completion: ChatCompletionResponse
    match tool_selection:
        case "auto":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                response_format=response_format,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"] or "",
                            "parameters": tool["parameters"],
                        },
                    }
                    for tool in tools
                ]
                if tools
                else None,
                tool_choice="auto",
            )

        case "none":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                response_format=response_format,
                tools=[],
                tool_choice="none",
            )

        case "required":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                response_format=response_format,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"] or "",
                            "parameters": tool["parameters"],
                        },
                    }
                    for tool in tools
                ]
                if tools
                else None,
                tool_choice="any",
            )

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                response_format=response_format,
                tools=[  # mistral can't be suggested with concrete tool
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"] or "",
                            "parameters": tool["parameters"],
                        },
                    }
                ],
                tool_choice="any",
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
        raise MistralException("Invalid Mistral completion - missing messages!", completion)

    completion_message: ChatMessageResponse = completion.choices[0].message

    if (tool_calls := completion_message.tool_calls) and (tools := tools):
        ctx.record(ResultTrace.of(tool_calls))

        return LMMToolRequests(
            requests=[
                LMMToolRequest(
                    identifier=call.id,
                    tool=call.function.name,
                    arguments=json.loads(call.function.arguments)
                    if isinstance(call.function.arguments, str)
                    else call.function.arguments,
                )
                for call in tool_calls
            ]
        )

    elif message := completion_message.content:
        ctx.record(ResultTrace.of(message))
        match message:
            case str(content):
                return LMMCompletion.of(content)

            # API docs say that it can be only a string in response
            # however library model allows list as well
            case other:
                return LMMCompletion.of(*other)

    else:
        raise MistralException("Invalid Mistral completion", completion)
