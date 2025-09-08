from collections.abc import AsyncGenerator, Coroutine, Iterable, Mapping, Sequence
from typing import Any, Literal, cast, overload

from haiway import META_EMPTY, MISSING, ObservabilityLevel, asynchronous, ctx, unwrap_missing

from draive.bedrock.api import BedrockAPI
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.models import (
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageContent,
    ChatMessageImage,
    ChatMessageText,
    ChatTool,
)
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelReasoning,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolResponse,
    ModelToolsDeclaration,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.multimodal import (
    MediaData,
    MetaContent,
    Multimodal,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)

__all__ = ("BedrockConverse",)


class BedrockConverse(BedrockAPI):
    def generative_model(self) -> GenerativeModel:
        return GenerativeModel(generating=self.completion)

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: Literal[False] = False,
        config: BedrockChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> Coroutine[None, None, ModelOutput]: ...

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: Literal[True],
        config: BedrockChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: bool = False,
        config: BedrockChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]:
        if stream:
            return self._completion_stream(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config or ctx.state(BedrockChatConfig),
                prefill=prefill,
                **extra,
            )

        return self._completion(
            instructions=instructions,
            context=context,
            tools=tools,
            output=output,
            config=config or ctx.state(BedrockChatConfig),
            prefill=prefill,
            **extra,
        )

    async def _completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        config: BedrockChatConfig,
        prefill: Multimodal | None,
        **extra: Any,
    ) -> ModelOutput:
        async with ctx.scope("model.completion"):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "model.provider": "bedrock",
                    "model.name": config.model,
                    "model.instructions": instructions,
                    "model.tools": [tool["name"] for tool in tools.specifications],
                    "model.tool_selection": tools.selection,
                    "model.context": [element.to_str() for element in context],
                    "model.temperature": config.temperature,
                    "model.stop_sequences": config.stop_sequences,
                    "model.max_output_tokens": config.max_output_tokens,
                    "model.output": str(output),
                },
            )

            # Build messages array from context
            messages: list[ChatMessage]

            if prefill is not None:
                messages = [
                    *_context_messages(context),
                    {
                        "role": "assistant",
                        "content": list(_convert_content(MultimodalContent.of(prefill).parts)),
                    },
                ]

            elif output == "json" or isinstance(output, type):
                messages = [
                    *_context_messages(context),
                    {
                        "role": "assistant",
                        "content": [{"text": "{"}],
                    },
                ]

            else:
                messages = list(_context_messages(context))

            tool_config: dict[str, Any] | None = _tools_as_tool_config(
                tools.specifications,
                tool_selection=tools.selection,
            )

            # Prepare and execute Converse request
            parameters: dict[str, Any] = {
                "modelId": config.model,
                "messages": messages,
                "inferenceConfig": {
                    "temperature": config.temperature,
                },
            }
            if instructions:
                parameters["system"] = [{"text": instructions}]
            if tool_config:
                parameters["toolConfig"] = tool_config
            if config.max_output_tokens is not MISSING:
                parameters["inferenceConfig"]["maxTokens"] = config.max_output_tokens
            if config.top_p is not MISSING:
                parameters["inferenceConfig"]["topP"] = config.top_p
            if config.stop_sequences:
                parameters["inferenceConfig"]["stopSequences"] = config.stop_sequences

            completion: ChatCompletionResponse = await self._converse(parameters)

            ctx.record(
                ObservabilityLevel.INFO,
                metric="model.input_tokens",
                value=completion["usage"]["inputTokens"],
                unit="tokens",
                kind="counter",
                attributes={
                    "model.provider": "bedrock",
                    "model.name": config.model,
                },
            )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="model.output_tokens",
                value=completion["usage"]["outputTokens"],
                unit="tokens",
                kind="counter",
                attributes={
                    "model.provider": "bedrock",
                    "model.name": config.model,
                },
            )

            # Convert output content preserving order of content and tool requests
            output_blocks = list(
                _completion_as_output_content(completion["output"]["message"]["content"])
            )

            stop_reason = completion["stopReason"]
            if stop_reason == "max_tokens":
                raise ModelOutputLimit(
                    provider="bedrock",
                    model=config.model,
                    max_output_tokens=unwrap_missing(config.max_output_tokens, default=0),
                    content=output_blocks,
                )

            if stop_reason in ("guardrail_intervened", "content_filtered"):
                from draive.models import ModelOutputFailed

                raise ModelOutputFailed(
                    provider="bedrock",
                    model=config.model,
                    reason=stop_reason,
                )

            return ModelOutput.of(
                *output_blocks,
                meta={
                    "model": config.model,
                    "stop_reason": stop_reason,
                },
            )

    async def _completion_stream(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        config: BedrockChatConfig,
        prefill: Multimodal | None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        async with ctx.scope("model.completion.stream"):
            ctx.log_warning(
                "Bedrock completion streaming is not supported yet, using regular response instead."
            )
            model_output: ModelOutput = await self._completion(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config,
                prefill=prefill,
                **extra,
            )

            for block in model_output.blocks:
                if isinstance(block, MultimodalContent):
                    for part in block.parts:
                        yield part

                else:
                    assert isinstance(block, ModelReasoning | ModelToolRequest)  # nosec: B101
                    yield block

    @asynchronous
    def _converse(
        self,
        parameters: dict[str, Any],
    ) -> ChatCompletionResponse:
        return self._client.converse(**parameters)


def _context_messages(  # noqa: C901, PLR0912
    context: ModelContext,
) -> Iterable[ChatMessage]:
    role: Literal["user", "assistant"] = "user"
    content: list[ChatMessageContent] = []

    def flush(
        current_role: Literal["user", "assistant"],
    ) -> ChatMessage | None:
        nonlocal content
        if not content:
            return None

        message: ChatMessage = {
            "role": current_role,
            "content": content,
        }
        content = []
        return message

    for element in context:
        if isinstance(element, ModelInput):
            if role != "user":
                if message := flush(role):
                    yield message

                role = "user"

            for block in element.blocks:
                if isinstance(block, MultimodalContent):
                    content.extend(_convert_content(block.parts))

                else:
                    assert isinstance(block, ModelToolResponse)  # nosec: B101
                    # tool response -> toolResult
                    content.append(
                        {
                            "toolResult": {
                                "toolUseId": block.identifier,
                                "content": cast(
                                    list[ChatMessageText | ChatMessageImage],
                                    list(_convert_content(block.content.parts)),
                                ),
                                "status": "error" if block.handling == "error" else "success",
                            }
                        }
                    )

        elif isinstance(element, ModelOutput):
            if role != "assistant":
                if message := flush(role):
                    yield message

                role = "assistant"

            for block in element.blocks:
                if isinstance(block, MultimodalContent):
                    content.extend(_convert_content(block.parts))

                elif isinstance(block, ModelReasoning):
                    continue  # skip reasoning

                else:
                    assert isinstance(block, ModelToolRequest)  # nosec: B101
                    # tool request -> toolUse
                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": block.identifier,
                                "name": block.tool,
                                "input": block.arguments,
                            }
                        }
                    )

        else:
            raise ValueError(f"Unsupported model context element: {type(element).__name__}")

    if message := flush(role):
        yield message


def _convert_content(
    parts: Sequence[MultimodalContentElement],
) -> Iterable[ChatMessageContent]:
    for part in parts:
        if isinstance(part, TextContent):
            yield {"text": part.text}

        elif isinstance(part, MediaData):
            if part.kind != "image":
                raise ValueError(f"Unsupported message content kind: {part.kind}")

            if part.media == "image/png":
                fmt = "png"

            elif part.media == "image/jpeg":
                fmt = "jpeg"

            elif part.media == "image/gif":
                fmt = "gif"

            else:
                raise ValueError(f"Unsupported message content kind: {part.kind}")

            yield {
                "image": {
                    "format": fmt,
                    "source": {
                        "bytes": part.data,
                    },
                },
            }

        elif isinstance(part, MetaContent):
            if part.category == "transcript" and part.content:
                yield {"text": part.content.to_str()}

            else:
                continue

        else:
            yield {"text": part.to_json()}


def _convert_tool(tool: ModelToolSpecification) -> ChatTool:
    return {
        "name": tool["name"],
        "description": tool["description"] or "",
        "inputSchema": {"json": tool["parameters"]},
    }


def _tools_as_tool_config(
    tools: Iterable[ModelToolSpecification] | None,
    /,
    *,
    tool_selection: ModelToolsSelection,
) -> dict[str, Any] | None:
    toolChoice: dict[str, Any]
    if tool_selection == "auto":
        toolChoice = {"auto": {}}

    elif tool_selection == "required":
        toolChoice = {"any": {}}

    elif tool_selection == "none":
        return None

    else:
        toolChoice = {
            "tool": {
                "name": cast(str, tool_selection),
            },
        }

    tools_list = [{"toolSpec": _convert_tool(tool)} for tool in tools or ()]
    if not tools_list:
        return None

    return {"tools": tools_list, "toolChoice": toolChoice}


def _completion_as_output_content(
    completion: Iterable[Mapping[str, Any]],
    /,
) -> Iterable[MultimodalContent | ModelToolRequest]:
    accumulator: list[MultimodalContentElement] = []
    for block in completion:
        match block:
            case {"text": str() as text}:
                accumulator.append(TextContent(text=text))

            case {
                "image": {
                    "format": str() as data_format,
                    "source": {"bytes": bytes() as data},
                }
            }:
                media_type: Any
                if data_format == "png":
                    media_type = "image/png"

                elif data_format == "jpeg":
                    media_type = "image/jpeg"

                elif data_format == "gif":
                    media_type = "image/gif"

                else:
                    raise ValueError(f"Unsupported output image format: {data_format}")

                accumulator.append(MediaData.of(data, media=media_type))

            case {
                "toolUse": {
                    "toolUseId": str() as identifier,
                    "name": str() as tool,
                    "input": arguments,
                }
            }:
                if accumulator:
                    yield MultimodalContent.of(*accumulator)
                    accumulator = []

                yield ModelToolRequest(
                    identifier=identifier,
                    tool=tool,
                    arguments=cast(Mapping[str, Any], arguments),
                    meta=META_EMPTY,
                )

            case _:
                continue

    if accumulator:
        yield MultimodalContent.of(*accumulator)
