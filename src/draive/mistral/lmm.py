import json
from collections.abc import Iterable
from typing import Any, Literal, cast

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInvocation,
    LMMOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolSelection,
    ToolSpecification,
)
from draive.metrics import TokenUsage
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.mistral.models import ChatCompletionResponse, ChatMessage, ChatMessageResponse
from draive.mistral.types import MistralException
from draive.multimodal import MultimodalContent
from draive.parameters import ParametersSpecification

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
        context: Iterable[LMMContextElement],
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        prefill: MultimodalContent | None,
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

            match output:
                case "auto" | "text":
                    config = config.updated(response_format={"type": "text"})

                case _:
                    if tools:
                        ctx.log_warning(
                            "Attempting to use Mistral in JSON mode with tools which is not"
                            " supported. Using text mode instead..."
                        )
                        config = config.updated(response_format={"type": "text"})

                    else:
                        config = config.updated(response_format={"type": "json_object"})

            if prefill:
                context = [*context, LMMCompletion.of(prefill)]

            messages: list[ChatMessage] = [
                _convert_context_element(element=element) for element in context
            ]

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
                tools=tools,
                tool_selection=tool_selection,
            )

    return LMMInvocation(invoke=lmm_invocation)


def _convert_context_element(
    element: LMMContextElement,
) -> ChatMessage:
    match element:
        case LMMInput() as input:
            return ChatMessage(
                role="user",
                content=input.content.as_string(),
            )

        case LMMCompletion() as completion:
            return ChatMessage(
                role="assistant",
                content=completion.content.as_string(),
            )

        case LMMToolRequests() as tool_requests:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": request.identifier,
                        "function": {
                            "name": request.tool,
                            "arguments": json.dumps(request.arguments),
                        },
                    }
                    for request in tool_requests.requests
                ],
            )

        case LMMToolResponse() as tool_response:
            return ChatMessage(
                role="tool",
                name=tool_response.tool,
                content=tool_response.content.as_string(),
            )


async def _chat_completion(
    *,
    client: MistralClient,
    config: MistralChatConfig,
    messages: list[ChatMessage],
    tools: Iterable[ToolSpecification] | None,
    tool_selection: LMMToolSelection,
) -> LMMOutput:
    completion: ChatCompletionResponse
    match tool_selection:
        case "auto":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[dict[str, object]],
                    tools,
                ),
                tool_choice="auto",
            )

        case "none":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=[],
                tool_choice="none",
            )

        case "required":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[dict[str, object]],
                    tools,
                ),
                tool_choice="any",
            )

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[dict[str, object]],
                    [tool],  # mistral can't be suggested with concrete tool
                ),
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
