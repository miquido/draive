"""Ollama chat adapter for GenerativeModel with tools and streaming."""

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Literal, cast
from uuid import uuid4

from haiway import Meta, State, as_list, ctx
from ollama import ChatResponse, Image, Message, Options, Tool

from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputSelection,
    ModelOutputStream,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.models.metrics import record_model_invocation, record_usage_metrics
from draive.multimodal import Multimodal, MultimodalContent, TextContent
from draive.multimodal.content import MultimodalContentPart
from draive.ollama.api import OllamaAPI
from draive.ollama.config import OllamaChatConfig
from draive.ollama.utils import unwrap_missing
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OllamaChat",)


class OllamaChat(OllamaAPI):
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: OllamaChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> ModelOutputStream:
        return self._completion_stream(
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
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: OllamaChatConfig,
        prefill: Multimodal | None,
        **extra: Any,
    ) -> ModelOutput:
        async with ctx.scope("model.invocation"):
            record_model_invocation(
                provider="ollama",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
            )

            messages: list[Message] = list(
                _context_messages(
                    instructions=instructions,
                    context=context,
                )
            )

            if prefill:
                messages.append(_assistant_message_from_content(MultimodalContent.of(prefill)))

            elif output == "json" or isinstance(output, type):
                messages.append(_assistant_message_from_content(MultimodalContent.of("{")))

            try:
                completion: ChatResponse = await self._client.chat(  # pyright: ignore[reportUnknownMemberType]
                    model=config.model,
                    messages=messages,
                    format=_response_format(output),
                    tools=_tools_as_tool_config(
                        tools.specification,
                        tool_selection=tools.selection,
                    ),
                    options=Options(
                        temperature=unwrap_missing(config.temperature),
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

                blocks.extend(
                    _tool_calls_to_requests(
                        completion.message.tool_calls,
                        model=config.model,
                    )
                )
                record_usage_metrics(
                    provider="ollama",
                    model=config.model,
                    input_tokens=completion.prompt_eval_count,
                    output_tokens=completion.eval_count,
                )

                return ModelOutput.of(
                    *blocks,
                    meta={
                        "model": config.model,
                    },
                )

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="ollama",
                    model=config.model,
                    reason=str(exc),
                ) from exc

    async def _completion_stream(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: OllamaChatConfig,
        prefill: Multimodal | None,
        **extra: Any,
    ) -> ModelOutputStream:
        async with ctx.scope("ollama.chat"):
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

            for block in model_output.output:
                if isinstance(block, MultimodalContent):
                    for part in block.parts:
                        yield part

                else:
                    assert isinstance(block, ModelToolRequest | ModelReasoningChunk)  # nosec: B101
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
            ctype: str | None = cast(dict[str, Any], chunk).get("type")
            if ctype == "text":
                text: Any = cast(dict[str, Any], chunk).get("text")
                if isinstance(text, str) and text:
                    return TextContent.of(text)

            if ctype in ("image", "image_url"):
                image: dict[str, Any] = (
                    cast(dict[str, Any], chunk).get("image")
                    or cast(dict[str, Any], chunk).get("image_url")
                    or {}
                )
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
    *,
    instructions: ModelInstructions,
    context: ModelContext,
) -> Iterable[Message]:
    if instructions:
        yield Message(
            role="system",
            content=instructions,
        )

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

            if responses := element.tool_responses:
                # Include any tool responses that follow the user message
                for tool_resp in responses:
                    yield Message(
                        role="tool",
                        tool_name=tool_resp.tool,
                        content=tool_resp.result.without_resources().to_str(),
                    )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
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
                    for request in element.tool_requests
                ],
            )


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


def _tool_call_arguments(
    arguments: object,
    /,
    *,
    model: str,
) -> dict[str, Any]:
    if isinstance(arguments, Mapping):
        return cast(dict[str, Any], arguments)

    if isinstance(arguments, str):
        try:
            loaded: object = json.loads(arguments)
        except Exception as exc:
            raise ModelOutputInvalid(
                provider="ollama",
                model=model,
                reason=f"Tool arguments decoding error - {type(exc).__name__}: {exc}",
            ) from exc

        if isinstance(loaded, Mapping):
            return cast(dict[str, Any], loaded)

        raise ModelOutputInvalid(
            provider="ollama",
            model=model,
            reason="Tool arguments should be a JSON object",
        )

    raise ModelOutputInvalid(
        provider="ollama",
        model=model,
        reason="Tool arguments should be a mapping or JSON object string",
    )


def _tool_calls_to_requests(
    tool_calls: Sequence[Message.ToolCall] | None,
    /,
    *,
    model: str,
) -> list[ModelToolRequest]:
    if not tool_calls:
        return []

    return [
        ModelToolRequest(
            identifier=str(uuid4()),  # ollama does not return an id
            tool=call.function.name,
            arguments=_tool_call_arguments(
                call.function.arguments,
                model=model,
            ),
            meta=Meta.empty,
        )
        for call in tool_calls
    ]


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

    # specific tool suggestion is not supported by Ollama
    return tools_list


def _schema_for_ollama(output: type[State]) -> dict[str, Any] | None:
    normalized_schema, changed = _normalize_schema_for_ollama(output.__SPECIFICATION__)
    if normalized_schema is None:
        return None

    if changed:
        ctx.log_debug(
            f"ollama schema normalized to remove unsupported keywords for {output.__name__}",
        )

    return normalized_schema


def _collapse_ollama_type_union(schema_types: Sequence[Any]) -> str | None:  # noqa: PLR0911
    allowed: Final[set[str]] = {"string", "number", "integer", "boolean", "null"}
    primitive_types: list[str] = []
    for schema_type in schema_types:
        if not isinstance(schema_type, str) or schema_type not in allowed:
            return None

        primitive_types.append(schema_type)

    if not primitive_types:
        return None

    distinct: set[str] = set(primitive_types)

    if distinct == {"null"}:
        return "null"

    if len(distinct) == 1:
        value = next(iter(distinct))
        return "number" if value == "integer" else value

    non_null: set[str] = distinct - {"null"}
    if not non_null:
        return "null"

    if non_null <= {"number", "integer"}:
        return "number"

    if "string" in non_null and non_null <= {"string", "number", "integer"}:
        return "string"

    if non_null == {"boolean"}:
        return "boolean"

    return None


def _normalize_schema_for_ollama(schema: Mapping[str, Any]) -> tuple[dict[str, Any] | None, bool]:  # noqa: C901, PLR0911, PLR0912, PLR0915
    # Remove metadata fields that Ollama rejects
    disallowed_root_keys: set[str] = {"$schema", "$id"}
    changed: bool = any(key in schema for key in disallowed_root_keys)

    schema_type: Any = schema.get("type")

    if isinstance(schema_type, Sequence) and not isinstance(schema_type, str | bytes):
        elements: list[Any] = as_list(cast(Sequence[Any], schema_type))
        collapsed: str | None = _collapse_ollama_type_union(elements)
        if collapsed is None:
            return None, changed

        if len(elements) != 1 or elements[0] != collapsed:
            changed = True

        schema_type = collapsed

    if schema_type == "object" or "properties" in schema or "required" in schema:
        normalized_properties: dict[str, Any] = {}
        properties_value: Any = schema.get("properties", {})
        if isinstance(properties_value, Mapping):
            for key, value in cast(Mapping[str, Any], properties_value).items():
                property_schema, property_changed = _normalize_schema_for_ollama(value)
                if property_schema is None:
                    return None, True

                normalized_properties[key] = property_schema
                changed = changed or property_changed

        elif "properties" in schema:
            changed = True

        normalized_schema: dict[str, Any] = {"type": "object"}
        if normalized_properties:
            normalized_schema["properties"] = normalized_properties

        required_value: Any = schema.get("required")
        if isinstance(required_value, Sequence) and not isinstance(required_value, str | bytes):
            filtered_required: list[str] = [
                name
                for name in cast(Sequence[Any], required_value)
                if name in normalized_properties
            ]
            if filtered_required:
                normalized_schema["required"] = filtered_required
            if len(filtered_required) != len(as_list(cast(Sequence[Any], required_value))):
                changed = True

        additional_properties: Any = schema.get("additionalProperties")
        if isinstance(additional_properties, bool):
            normalized_schema["additionalProperties"] = additional_properties
        elif isinstance(additional_properties, Mapping):
            # Ollama cannot handle schema-valued entries; drop but keep the object type.
            changed = True
        elif additional_properties is not None:
            changed = True

        if isinstance(schema.get("description"), str):
            normalized_schema["description"] = schema["description"]

        return normalized_schema, changed

    if schema_type == "array" or "items" in schema or "prefixItems" in schema:
        normalized_schema = {"type": "array"}

        if "items" in schema:
            item_schema, item_changed = _normalize_schema_for_ollama(schema["items"])
            if item_schema is not None:
                normalized_schema["items"] = item_schema
                changed = changed or item_changed
            else:
                changed = True

        elif "prefixItems" in schema:
            prefix_items: Any = schema.get("prefixItems")
            if (
                isinstance(prefix_items, Sequence)
                and not isinstance(prefix_items, str | bytes)
                and len(cast(Sequence[Any], prefix_items)) == 1
            ):
                item_schema, item_changed = _normalize_schema_for_ollama(
                    cast(Sequence[Any], prefix_items)[0]
                )
                if item_schema is not None:
                    normalized_schema["items"] = item_schema
                    changed = True
                    changed = changed or item_changed
                else:
                    changed = True
            elif isinstance(prefix_items, Sequence) and not isinstance(prefix_items, str | bytes):
                changed = True
            else:
                changed = True

        if isinstance(schema.get("description"), str):
            normalized_schema["description"] = schema["description"]

        if "minItems" in schema or "maxItems" in schema or "uniqueItems" in schema:
            changed = True

        return normalized_schema, changed

    if schema_type in {"string", "number", "integer", "boolean", "null"}:
        normalized_schema = {"type": schema_type}

        enum_value: Any = schema.get("enum")
        if isinstance(enum_value, Sequence) and not isinstance(enum_value, str | bytes):
            normalized_schema["enum"] = list(cast(Sequence[Any], enum_value))

        if schema_type == "string" and isinstance(schema.get("format"), str):
            normalized_schema["format"] = schema["format"]

        if isinstance(schema.get("description"), str):
            normalized_schema["description"] = schema["description"]

        return normalized_schema, changed

    if "enum" in schema and schema_type is None:
        enum_value = schema["enum"]
        if (
            isinstance(enum_value, Sequence)
            and not isinstance(enum_value, str | bytes)
            and enum_value
        ):
            inferred_type: Any = cast(Any, type(cast(Sequence[Any], enum_value)[0]))
            if inferred_type in {str, int, float, bool}:
                mapped_type = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                }[inferred_type]
                normalized_schema = {
                    "type": mapped_type,
                    "enum": list(cast(Sequence[Any], enum_value)),
                }
                return normalized_schema, True

    if "$ref" in schema or any(key in schema for key in ("oneOf", "anyOf", "allOf", "not")):
        return None, changed

    return None, changed


def _response_format(
    output: ModelOutputSelection,
) -> Literal["json"] | dict[str, Any] | None:
    # Explicit JSON output (no schema)
    if output == "json":
        return "json"

    # Structured output with DataModel schema
    if isinstance(output, type):
        if (schema := _schema_for_ollama(output)) is None:
            ctx.log_warning(
                f"ollama format fallback to plain json due to unsupported schema"
                f" constructs for {output.__name__}",
            )
            return "json"

        return schema

    return None
