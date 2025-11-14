import json
from collections.abc import AsyncGenerator, Coroutine, Generator, Iterable
from typing import Any, Literal, cast, overload
from uuid import uuid4

from haiway import META_EMPTY, as_list, ctx, unwrap_missing
from mistralai import ContentChunkTypedDict, ImageURLChunk, TextChunk, ThinkChunk, ToolTypedDict
from mistralai.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequestToolChoiceTypedDict,
    ChatCompletionResponse,
    CompletionEvent,
    CompletionResponseStreamChoice,
    ContentChunk,
    DeltaMessage,
    MessagesTypedDict,
    ResponseFormatTypedDict,
    ToolCall,
)
from mistralai.utils.eventstreaming import EventStreamAsync

from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralChatConfig
from draive.mistral.utils import unwrap_missing_to_none, unwrap_missing_to_unset
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelReasoning,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolsDeclaration,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.multimodal import (
    ArtifactContent,
    Multimodal,
    MultimodalContent,
    MultimodalContentPart,
    TextContent,
)
from draive.resources import ResourceContent, ResourceReference

__all__ = ("MistralCompletions",)


class MistralCompletions(MistralAPI):
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
        config: MistralChatConfig | None = None,
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
        config: MistralChatConfig | None = None,
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
        config: MistralChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]:
        if stream:
            return self._completion_stream(
                instructions=instructions,
                tools=tools,
                context=context,
                output=output,
                config=config or ctx.state(MistralChatConfig),
                prefill=prefill,
                **extra,
            )

        return self._completion(
            instructions=instructions,
            tools=tools,
            context=context,
            output=output,
            config=config or ctx.state(MistralChatConfig),
            prefill=prefill,
        )

    async def _completion_stream(  # noqa: C901, PLR0912
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        config: MistralChatConfig,
        prefill: Multimodal | None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        ctx.record_info(
            attributes={
                "model.provider": "mistral",
                "model.name": config.model,
                "model.temperature": config.temperature,
                "model.max_output_tokens": config.max_output_tokens,
                "model.output": str(output),
                "model.tools.count": len(tools.specifications),
                "model.tools.selection": tools.selection,
                "model.stream": True,
            },
        )
        ctx.record_debug(
            attributes={
                "model.instructions": instructions,
                "model.tools": [tool.name for tool in tools.specifications],
                "model.context": [element.to_str() for element in context],
            },
        )
        messages: list[MessagesTypedDict] = _build_messages(
            context=context,
            instructions=instructions,
            prefill=prefill,
        )

        tool_choice: ChatCompletionRequestToolChoiceTypedDict
        tools_list: list[ToolTypedDict]
        tool_choice, tools_list = _tools_as_tool_config(
            tools.specifications,
            tool_selection=tools.selection,
        )

        stream: EventStreamAsync[CompletionEvent] = await self._client.chat.stream_async(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            top_p=unwrap_missing_to_none(config.top_p),
            max_tokens=unwrap_missing_to_unset(config.max_output_tokens),
            stop=as_list(unwrap_missing_to_none(config.stop_sequences)),
            random_seed=unwrap_missing_to_unset(config.seed),
            response_format=_response_format(output),
            tools=tools_list,
            tool_choice=tool_choice,
            stream=True,
        )

        # Accumulate tool_calls from parts
        accumulated_tool_calls: list[ToolCall] = []

        async for event in stream:
            if usage := event.data.usage:
                ctx.record_info(
                    metric="model.input_tokens",
                    value=usage.prompt_tokens or 0,
                    unit="tokens",
                    kind="counter",
                    attributes={
                        "model.provider": "mistral",
                        "model.name": event.data.model,
                    },
                )
                ctx.record_info(
                    metric="model.output_tokens",
                    value=usage.completion_tokens or 0,
                    unit="tokens",
                    kind="counter",
                    attributes={
                        "model.provider": "mistral",
                        "model.name": event.data.model,
                    },
                )

            elif not event.data.choices:  # allow empty with usage data
                raise ModelOutputFailed(
                    provider="mistral",
                    model=config.model,
                    reason="Invalid completion: missing delta choices",
                )

            completion_choice: CompletionResponseStreamChoice = event.data.choices[0]

            completion_delta: DeltaMessage = completion_choice.delta
            if content := completion_delta.content:
                if isinstance(content, str):
                    yield TextContent.of(content)

                else:
                    for part in content:
                        yield _content_chunk_as_content_element(part)

            if tool_calls := completion_delta.tool_calls:
                if not accumulated_tool_calls:
                    accumulated_tool_calls = sorted(
                        tool_calls,
                        key=lambda call: call.index or 0,
                    )

                else:
                    for tool_call in tool_calls:
                        if tool_call.index is None:
                            raise ModelOutputFailed(
                                provider="mistral",
                                model=config.model,
                                reason="Invalid completion: missing tool call index",
                            )

                        index: int = tool_call.index

                        # "null" is a dafault value...
                        if tool_call.id and tool_call.id != "null":
                            accumulated_tool_calls[index].id = tool_call.id

                        if tool_call.function.name:
                            accumulated_tool_calls[index].function.name += tool_call.function.name

                        if isinstance(tool_call.function.arguments, str):
                            assert isinstance(  # nosec: B101
                                accumulated_tool_calls[index].function.arguments,
                                str,
                            )
                            accumulated_tool_calls[  # pyright: ignore[reportOperatorIssue]
                                index
                            ].function.arguments += tool_call.function.arguments

                        else:
                            assert isinstance(  # nosec: B101
                                accumulated_tool_calls[index].function.arguments,
                                dict,
                            )
                            accumulated_tool_calls[index].function.arguments = {
                                **cast(
                                    dict[str, Any],
                                    accumulated_tool_calls[index].function.arguments,
                                ),
                                **tool_call.function.arguments,
                            }

            if completion_choice.finish_reason == "error":
                raise ModelOutputFailed(
                    provider="mistral",
                    model=config.model,
                    reason=completion_choice.finish_reason,
                )

            if completion_choice.finish_reason in ("length", "model_length"):
                raise ModelOutputLimit(
                    provider="mistral",
                    model=config.model,
                    max_output_tokens=unwrap_missing(config.max_output_tokens, default=0),
                    content=(),  # already streamed
                )

            # finally send tool calls if any
            for call in accumulated_tool_calls:
                if not call.function:
                    continue  # skip partial calls

                if not call.function.name:
                    continue  # skip calls with missing names

                yield ModelToolRequest(
                    identifier=call.id or uuid4().hex,
                    tool=call.function.name,
                    arguments=(
                        json.loads(call.function.arguments)
                        if isinstance(call.function.arguments, str)
                        else call.function.arguments
                    ),
                    meta=META_EMPTY,
                )

    async def _completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        config: MistralChatConfig,
        prefill: Multimodal | None,
    ) -> ModelOutput:
        async with ctx.scope("model.completion"):
            ctx.record_info(
                attributes={
                    "model.provider": "mistral",
                    "model.name": config.model,
                    "model.temperature": config.temperature,
                    "model.max_output_tokens": config.max_output_tokens,
                    "model.output": str(output),
                    "model.tools.count": len(tools.specifications),
                    "model.tools.selection": tools.selection,
                    "model.stream": False,
                },
            )
            ctx.record_debug(
                attributes={
                    "model.instructions": instructions,
                    "model.tools": [tool.name for tool in tools.specifications],
                    "model.context": [element.to_str() for element in context],
                },
            )

            messages: list[MessagesTypedDict] = _build_messages(
                context=context,
                instructions=instructions,
                prefill=prefill,
            )

            tool_choice: ChatCompletionRequestToolChoiceTypedDict
            tools_list: list[ToolTypedDict]
            tool_choice, tools_list = _tools_as_tool_config(
                tools.specifications,
                tool_selection=tools.selection,
            )

            completion: ChatCompletionResponse = await self._client.chat.complete_async(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                top_p=unwrap_missing_to_none(config.top_p),
                max_tokens=unwrap_missing_to_unset(config.max_output_tokens),
                stop=as_list(unwrap_missing_to_none(config.stop_sequences)),
                random_seed=unwrap_missing_to_unset(config.seed),
                response_format=_response_format(output),
                tools=tools_list,
                tool_choice=tool_choice,
                stream=False,
            )

            _record_usage_metrics(completion)

            if not completion.choices:
                raise ModelOutputFailed(
                    provider="mistral",
                    model=config.model,
                    reason="Invalid completion: missing choices",
                )

            choice: ChatCompletionChoice = completion.choices[0]

            if choice.finish_reason == "error":
                raise ModelOutputFailed(
                    provider="mistral",
                    model=config.model,
                    reason=choice.finish_reason,
                )

            if choice.finish_reason in ("length", "model_length"):
                raise ModelOutputLimit(
                    provider="mistral",
                    model=config.model,
                    max_output_tokens=unwrap_missing(config.max_output_tokens, default=0),
                    content=tuple(_message_to_blocks(choice.message)),
                )

            return ModelOutput.of(
                *_message_to_blocks(choice.message),
                meta={
                    "identifier": completion.id,
                    "model": config.model,
                    "finish_reason": choice.finish_reason,
                },
            )


def _context_messages(
    context: ModelContext,
) -> Iterable[MessagesTypedDict]:
    for element in context:
        if isinstance(element, ModelInput):
            if user_content := element.content:
                yield {
                    "role": "user",
                    "content": list(_content_chunks(user_content.parts)),
                }

            # Provide tool responses as separate tool messages expected by Mistral
            for tool_response in element.tools:
                yield {
                    "role": "tool",
                    "tool_call_id": tool_response.identifier,
                    "name": tool_response.tool,
                    "content": list(_content_chunks(tool_response.content.parts)),
                }

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            for block in element.blocks:
                if isinstance(block, MultimodalContent):
                    yield {
                        "role": "assistant",
                        "content": list(_content_chunks(block.parts)),
                    }

                elif isinstance(block, ModelReasoning):
                    continue  # skip reasoning

                else:
                    assert isinstance(block, ModelToolRequest)  # nosec: B101
                    yield {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": block.identifier,
                                "function": {
                                    "name": block.tool,
                                    "arguments": json.dumps(block.arguments),
                                },
                            }
                        ],
                    }


def _build_messages(
    *,
    instructions: ModelInstructions,
    context: ModelContext,
    prefill: Multimodal | None,
) -> list[MessagesTypedDict]:
    messages: list[MessagesTypedDict]
    if instructions:
        messages = [
            {
                "role": "system",
                "content": instructions,
            },
            *_context_messages(context),
        ]

    else:
        messages = list(_context_messages(context))

    if prefill is not None:
        messages.append(
            cast(
                MessagesTypedDict,
                {
                    "role": "assistant",
                    "content": list(_content_chunks(MultimodalContent.of(prefill).parts)),
                },
            )
        )

    return messages


def _content_chunks(
    parts: Iterable[MultimodalContentPart],
) -> Iterable[ContentChunkTypedDict]:
    for element in parts:
        if isinstance(element, TextContent):
            yield {
                "type": "text",
                "text": element.text,
            }

        elif isinstance(element, ResourceReference):
            if not (element.mime_type or "").startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {element.mime_type}")

            yield {
                "type": "image_url",
                "image_url": {
                    "url": element.uri,
                },
            }

        elif isinstance(element, ResourceContent):
            if not (element.mime_type or "").startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {element.mime_type}")

            yield {
                "type": "image_url",
                "image_url": {
                    # ResourceContent.to_data_uri() returns a proper data URI
                    "url": element.to_data_uri(),
                },
            }

        else:
            assert isinstance(element, ArtifactContent)  # nosec: B101
            if element.hidden:
                continue

            yield {
                "type": "text",
                "text": element.artifact.to_str(),
            }


def _content_chunk_as_content_element(
    chunk: ContentChunk,
) -> MultimodalContentPart:
    if isinstance(chunk, TextChunk):
        return TextContent.of(chunk.text)

    elif isinstance(chunk, ThinkChunk):
        return ArtifactContent.of(
            ModelReasoning.of(
                *(part.text for part in chunk.thinking if isinstance(part, TextChunk))
            ),
            category="reasoning",
            hidden=True,
        )

    elif isinstance(chunk, ImageURLChunk):
        return ResourceReference.of(
            chunk.image_url if isinstance(chunk.image_url, str) else chunk.image_url.url,
            mime_type="image/png",
        )

    else:
        raise ValueError(f"Unsupported content chunk: {type(chunk)}")


def _record_usage_metrics(
    completion: ChatCompletionResponse,
) -> None:
    if usage := completion.usage:
        ctx.record_info(
            metric="model.input_tokens",
            value=usage.prompt_tokens or 0,
            unit="tokens",
            kind="counter",
            attributes={"model.provider": "mistral", "model.name": completion.model},
        )
        ctx.record_info(
            metric="model.output_tokens",
            value=usage.completion_tokens or 0,
            unit="tokens",
            kind="counter",
            attributes={"model.provider": "mistral", "model.name": completion.model},
        )


def _message_to_blocks(
    message: AssistantMessage,
) -> Generator[MultimodalContent | ModelToolRequest]:
    if content := message.content:
        if isinstance(content, str):
            yield MultimodalContent.of(TextContent(text=content))

        else:
            yield MultimodalContent.of(
                *(_content_chunk_as_content_element(chunk) for chunk in content)
            )

    if tool_calls := message.tool_calls:
        for call in tool_calls:
            yield ModelToolRequest(
                identifier=call.id or "",
                tool=call.function.name,
                arguments=(
                    json.loads(call.function.arguments)
                    if isinstance(call.function.arguments, str)
                    else call.function.arguments
                ),
                meta=META_EMPTY,
            )


def _tool_specification_as_tool(
    tool: ModelToolSpecification,
) -> ToolTypedDict:
    # Mistral requires a valid JSON schema object for parameters; provide a minimal placeholder
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": cast(dict[str, Any], tool.parameters)
            or {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _tools_as_tool_config(
    tools: Iterable[ModelToolSpecification] | None,
    /,
    *,
    tool_selection: ModelToolsSelection,
) -> tuple[ChatCompletionRequestToolChoiceTypedDict, list[ToolTypedDict]]:
    tools_list: list[ToolTypedDict] = [_tool_specification_as_tool(tool) for tool in (tools or [])]
    if not tools_list:
        return ("none", tools_list)

    if tool_selection == "auto":
        return ("auto", tools_list)

    if tool_selection == "none":
        return ("none", [])

    if tool_selection == "required":
        return ("any", tools_list)

    return (
        {
            "type": "function",
            "function": {
                "name": tool_selection,
            },
        },
        tools_list,
    )


def _response_format(
    output: ModelOutputSelection,
) -> ResponseFormatTypedDict | None:
    if output == "json":
        return cast(ResponseFormatTypedDict, {"type": "json_object"})

    if isinstance(output, type):
        # Structured output with DataModel schema

        return cast(
            ResponseFormatTypedDict,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": output.__name__,
                    "schema": output.__SPECIFICATION__,
                },
            },
        )

    return None
