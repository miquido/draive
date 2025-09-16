"""Ollama chat adapter for GenerativeModel with tools and streaming."""

import json
from collections.abc import AsyncGenerator, Coroutine, Iterable
from typing import Any, Literal, cast, overload
from uuid import uuid4

from haiway import META_EMPTY, ObservabilityLevel, ctx
from ollama import ChatResponse, Image, Message, Options, Tool

from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputSelection,
    ModelReasoning,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolsDeclaration,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.multimodal import Multimodal, MultimodalContent, TextContent
from draive.multimodal.content import MultimodalContentPart
from draive.ollama.api import OllamaAPI
from draive.ollama.config import OllamaChatConfig
from draive.ollama.utils import unwrap_missing
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OllamaChat",)


class OllamaChat(OllamaAPI):
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
        config: OllamaChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: object,
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
        config: OllamaChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: object,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: bool = False,
        config: OllamaChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]:
        if stream:
            return self._completion_stream(
                instructions=instructions,
                tools=tools,
                context=context,
                output=output,
                config=config or ctx.state(OllamaChatConfig),
                prefill=prefill,
                **extra,
            )

        return self._completion(
            instructions=instructions,
            tools=tools,
            context=context,
            output=output,
            config=config or ctx.state(OllamaChatConfig),
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
        config: OllamaChatConfig,
        prefill: Multimodal | None,
        **extra: Any,
    ) -> ModelOutput:
        async with ctx.scope("model.completion"):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "model.provider": "ollama",
                    "model.name": config.model,
                    "model.temperature": config.temperature,
                    "model.output": str(output),
                    "model.tools.count": len(tools.specifications),
                    "model.tools.selection": tools.selection,
                    "model.stream": False,
                },
            )
            ctx.record(
                ObservabilityLevel.DEBUG,
                attributes={
                    "model.instructions": instructions,
                    "model.tools": [tool.name for tool in tools.specifications],
                    "model.context": [element.to_str() for element in context],
                },
            )

            messages: list[Message] = list(_context_messages(context))

            if prefill:
                messages.append(_assistant_message_from_content(MultimodalContent.of(prefill)))

            elif output == "json" or isinstance(output, type):
                messages.append(_assistant_message_from_content(MultimodalContent.of("{")))

            completion: ChatResponse = await self._client.chat(
                model=config.model,
                messages=messages,
                format=_response_format(output),
                tools=_tools_as_tool_config(
                    tools.specifications,
                    tool_selection=tools.selection,
                ),
                options=Options(
                    temperature=config.temperature,
                    num_predict=unwrap_missing(config.max_output_tokens),
                    top_k=unwrap_missing(config.top_k),
                    top_p=unwrap_missing(config.top_p),
                    seed=unwrap_missing(config.seed),
                    stop=unwrap_missing(config.stop_sequences),
                ),
                stream=False,
            )

            blocks: list[ModelOutputBlock] = []
            # Convert message content into content and reasoning blocks
            blocks.extend(_message_to_blocks(completion.message))

            # Append tool requests if present
            if tool_calls := completion.message.tool_calls:
                blocks.extend(
                    [
                        ModelToolRequest(
                            identifier=uuid4().hex,  # ollama does not return an id
                            tool=call.function.name,
                            arguments=(
                                json.loads(call.function.arguments)
                                if isinstance(call.function.arguments, str)
                                else call.function.arguments
                            ),
                            meta=META_EMPTY,
                        )
                        for call in tool_calls
                    ],
                )

            return ModelOutput.of(
                *blocks,
                meta={
                    "model": config.model,
                },
            )

    async def _completion_stream(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        config: OllamaChatConfig,
        prefill: Multimodal | None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        async with ctx.scope("model.completion.stream"):
            ctx.log_warning(
                "ollama completion streaming is not supported yet, using regular response instead."
            )

            model_output: ModelOutput = await self._completion(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config,
                prefill=prefill,
            )

            for block in model_output.blocks:
                if isinstance(block, MultimodalContent):
                    for part in block.parts:
                        yield part

                elif isinstance(block, ModelReasoning | ModelToolRequest):
                    yield block


def _assistant_message_from_content(
    content: MultimodalContent,
) -> Message:
    return Message(
        role="assistant",
        content=content.without_resources().to_str(),
        images=[
            # Prefer URLs for references; use data URIs for inline content
            Image(value=image.uri)
            if isinstance(image, ResourceReference)
            else Image(value=image.to_data_uri())
            for image in content.images()
        ]
        or None,
    )


def _message_to_blocks(  # noqa: C901
    message: Message,
) -> list[MultimodalContent]:
    blocks: list[MultimodalContent] = []

    def flush(acc: list[MultimodalContentPart]) -> None:
        if acc:
            blocks.append(MultimodalContent.of(*acc))
            acc.clear()

    def _convert_chunk(
        chunk: object,
    ) -> MultimodalContentPart | None:
        if isinstance(chunk, dict):
            ctype = chunk.get("type")
            if ctype == "text":
                text = chunk.get("text")
                if isinstance(text, str) and text:
                    return TextContent.of(text)

            if ctype in ("image", "image_url"):
                image = chunk.get("image") or chunk.get("image_url") or {}
                url = image.get("url")
                if isinstance(url, str) and url:
                    return ResourceReference.of(url, mime_type="image/*")

        # Fallback: stringify any unknown object deterministically
        return TextContent.of(json.dumps(chunk, default=str))

    accumulator: list[MultimodalContentPart] = []
    content = message.content
    if isinstance(content, str) and content:
        accumulator.append(TextContent.of(content))

    elif isinstance(content, list):
        for chunk in content:
            converted = _convert_chunk(chunk)
            if isinstance(converted, TextContent | ResourceReference | ResourceContent):
                accumulator.append(converted)

    # Include images if present on message (some SDKs expose them separately)
    if getattr(message, "images", None):
        for img in message.images or []:
            if isinstance(img, bytes | bytearray):
                accumulator.append(ResourceContent.of(bytes(img), mime_type="image/*"))
            elif isinstance(img, str):
                accumulator.append(ResourceReference.of(img, mime_type="image/*"))

    flush(accumulator)
    return blocks


def _context_messages(
    context: ModelContext,
) -> Iterable[Message]:
    for element in context:
        if isinstance(element, ModelInput):
            if content := element.content:
                yield Message(
                    role="user",
                    content=content.without_resources().to_str(),
                    images=[
                        Image(value=image.uri)
                        if isinstance(image, ResourceReference)
                        else Image(value=image.to_data_uri())
                        for image in content.images()
                    ]
                    or None,
                )

            if responses := element.tools:
                # Include any tool responses that follow the user message
                for tool_resp in responses:
                    yield Message(
                        role="tool",
                        tool_name=tool_resp.tool,
                        content=tool_resp.content.without_resources().to_str(),
                    )

        elif isinstance(element, ModelOutput):
            content = element.content
            yield Message(
                role="assistant",
                content=content.without_resources().to_str(),
                images=[
                    Image(value=image.uri)
                    if isinstance(image, ResourceReference)
                    else Image(value=image.to_data_uri())
                    for image in content.images()
                ]
                or None,
                tool_calls=[
                    Message.ToolCall(
                        function=Message.ToolCall.Function(
                            name=request.tool,
                            arguments=cast(dict[str, Any], request.arguments),
                        ),
                    )
                    for request in element.tools
                ],
            )

        else:
            raise TypeError(f"Unsupported model context element: {type(element).__name__}")


def _tool_specification_as_tool(
    tool: ModelToolSpecification,
) -> Tool:
    return Tool(
        type="function",
        function=Tool.Function(
            name=tool.name,
            description=tool.description,
            parameters=(
                cast(Tool.Function.Parameters, tool.parameters)  # type: ignore[arg-type]
                if tool.parameters
                else {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                }
            ),
        ),
    )


def _tools_as_tool_config(
    tools: Iterable[ModelToolSpecification] | None,
    /,
    *,
    tool_selection: ModelToolsSelection,
) -> list[Tool] | None:
    tools_list: list[Tool] = [_tool_specification_as_tool(tool) for tool in (tools or [])]
    if not tools_list:
        return None

    if tool_selection == "auto":
        return tools_list

    if tool_selection == "none":
        return None

    if tool_selection == "required":
        # Ollama doesn't support hard-required tool selection
        return tools_list

    # suggestions not supported
    return tools_list


def _response_format(
    output: ModelOutputSelection,
) -> Literal["json"] | dict[str, Any] | None:
    # Explicit JSON output (no schema)
    if output == "json":
        return "json"

    # Structured output with DataModel schema
    if isinstance(output, type):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output.__name__,
                "schema": output.__PARAMETERS_SPECIFICATION__,
            },
        }

    return None
