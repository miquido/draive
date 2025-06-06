from collections.abc import AsyncIterator
from typing import Any, Literal, overload

from haiway import ObservabilityLevel, asynchronous, ctx

from draive.bedrock.api import BedrockAPI
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.lmm import (
    convert_context_element,
    output_as_response_declaration,
    tools_as_tool_config,
)
from draive.bedrock.models import ChatCompletionResponse, ChatMessage
from draive.bedrock.types import BedrockException
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMInstruction,
    LMMOutput,
    LMMOutputSelection,
    LMMStreamOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMTools,
)
from draive.multimodal import MediaData, MultimodalContent, MultimodalContentElement, TextContent


class BedrockLMMGeneration(BedrockAPI):
    def lmm(self) -> LMM:
        return LMM(completing=self.lmm_completion)

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        config: BedrockChatConfig | None = None,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: Literal[True],
        config: BedrockChatConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(  # noqa: C901, PLR0912
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: bool = False,
        config: BedrockChatConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        completion_config: BedrockChatConfig = config or ctx.state(BedrockChatConfig)
        tools = tools or LMMTools.none
        with ctx.scope("bedrock_lmm_completion", completion_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "lmm.provider": "bedrock",
                    "lmm.model": completion_config.model,
                    "lmm.temperature": completion_config.temperature,
                    "lmm.max_tokens": completion_config.max_tokens,
                    "lmm.tools": [tool["name"] for tool in tools.specifications],
                    "lmm.tool_selection": f"{tools.selection}",
                    "lmm.stream": stream,
                    "lmm.output": f"{output}",
                    "lmm.instruction": f"{instruction}",
                    "lmm.context": [element.to_str() for element in context],
                },
            )

            if stream:
                raise NotImplementedError("bedrock streaming is not implemented yet")

            output_decoder = output_as_response_declaration(output)

            messages: list[ChatMessage] = [convert_context_element(element) for element in context]

            tool_config = tools_as_tool_config(
                tools.specifications,
                tool_selection=tools.selection,
            )

            completion: ChatCompletionResponse = await self._chat_completion(
                config=completion_config,
                instruction=instruction,
                messages=messages,
                tool_config=tool_config,
            )

            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.input_tokens",
                value=completion["usage"]["inputTokens"],
                unit="tokens",
                attributes={"lmm.model": completion_config.model},
            )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.output_tokens",
                value=completion["usage"]["outputTokens"],
                unit="tokens",
                attributes={"lmm.model": completion_config.model},
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

                    ctx.record(
                        ObservabilityLevel.INFO,
                        event="lmm.completion",
                    )
                    return message_completion

                case "tool_use":
                    tools_completion = LMMToolRequests(requests=tool_calls)
                    ctx.record(
                        ObservabilityLevel.INFO,
                        event="lmm.tool_requests",
                        attributes={"lmm.tools": [call.tool for call in tool_calls]},
                    )
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
