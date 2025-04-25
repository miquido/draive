from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, overload

from haiway import ArgumentsTrace, ResultTrace, asynchronous, ctx

from draive.bedrock.api import BedrockAPI
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.lmm import (
    convert_context_element,
    output_as_response_declaration,
    tools_as_tool_config,
)
from draive.bedrock.models import ChatCompletionResponse, ChatMessage
from draive.bedrock.types import BedrockException
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMOutput,
    LMMOutputSelection,
    LMMStreamOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.metrics import TokenUsage
from draive.multimodal import MediaData, MultimodalContent, MultimodalContentElement, TextContent


class BedrockLMMGeneration(BedrockAPI):
    def lmm(self) -> LMM:
        return LMM(completing=self.lmm_completion)

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        config: BedrockChatConfig | None = None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        config: BedrockChatConfig | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(  # noqa: C901, PLR0912
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        config: BedrockChatConfig | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        with ctx.scope("bedrock_lmm_completion"):
            completion_config: BedrockChatConfig = config or ctx.state(BedrockChatConfig).updated(
                **extra
            )
            ctx.record(
                ArgumentsTrace.of(
                    config=completion_config,
                    instruction=instruction,
                    context=context,
                    tool_selection=tool_selection,
                    tools=tools,
                    output=output,
                    **extra,
                ),
            )

            if stream:
                raise NotImplementedError("bedrock streaming is not implemented yet")

            completion_config: BedrockChatConfig = ctx.state(BedrockChatConfig).updated(**extra)
            ctx.record(
                ArgumentsTrace.of(
                    config=completion_config,
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                ),
            )

            output_decoder = output_as_response_declaration(output)

            messages: list[ChatMessage] = [convert_context_element(element) for element in context]

            tool_config = tools_as_tool_config(tools, tool_selection=tool_selection)

            completion: ChatCompletionResponse = await self._chat_completion(
                config=completion_config,
                instruction=Instruction.formatted(instruction),
                messages=messages,
                tool_config=tool_config,
            )

            ctx.record(
                TokenUsage.for_model(
                    completion_config.model,
                    input_tokens=completion["usage"]["inputTokens"],
                    cached_tokens=None,
                    output_tokens=completion["usage"]["outputTokens"],
                ),
            )

            message_parts: list[MultimodalContentElement] = []
            tool_calls: list[LMMToolRequest] = []
            for part in completion["output"]["message"]["content"]:
                match part:
                    case {"text": str() as text}:
                        message_parts.append(TextContent(text=text))

                    case {
                        "image": {
                            "format": str() as data_format,
                            "source": {"bytes": bytes() as data},
                        }
                    }:
                        media_type: Any
                        match data_format:
                            case "png":
                                media_type = "image/png"

                            case "jpeg":
                                media_type = "image/jpeg"

                            case "gif":
                                media_type = "image/gif"

                            case _:  # pyright: ignore[reportUnnecessaryComparison]
                                media_type = "image"

                        message_parts.append(
                            MediaData.of(
                                data,
                                media=media_type,
                            )
                        )

                    case {
                        "toolUse": {
                            "toolUseId": str() as identifier,
                            "name": str() as tool,
                            "input": arguments,
                        }
                    }:
                        tool_calls.append(
                            LMMToolRequest(
                                identifier=identifier,
                                tool=tool,
                                arguments=arguments,
                            )
                        )

                    case _:
                        pass

            match completion["stopReason"]:
                case "end_turn" | "stop_sequence":
                    message_completion = LMMCompletion.of(
                        output_decoder(MultimodalContent.of(*message_parts))
                    )
                    ctx.record(ResultTrace.of(message_completion))
                    return message_completion

                case "tool_use":
                    tools_completion = LMMToolRequests(requests=tool_calls)
                    ctx.record(ResultTrace.of(tools_completion))
                    return tools_completion

                case _:
                    raise BedrockException("Invalid Bedrock response")

    @asynchronous
    def _chat_completion(
        self,
        *,
        config: BedrockChatConfig,
        instruction: str | None,
        messages: list[ChatMessage],
        tool_config: dict[str, Any] | None,
    ) -> ChatCompletionResponse:
        parameters: dict[str, Any] = {
            "modelId": config.model,
            "messages": messages,
            "inferenceConfig": {
                "temperature": config.temperature,
            },
        }

        if instruction:
            parameters["system"] = [
                {
                    "text": instruction,
                }
            ]

        if tool_config:
            parameters["toolConfig"] = tool_config
        if config.max_tokens:
            parameters["inferenceConfig"]["maxTokens"] = config.max_tokens
        if config.top_p:
            parameters["inferenceConfig"]["topP"] = config.top_p
        if config.stop_sequences:
            parameters["inferenceConfig"]["stopSequences"] = config.stop_sequences

        return self._client.converse(**parameters)
