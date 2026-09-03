"""Ollama chat adapter for GenerativeModel with tools and streaming."""

from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from typing import Any, Final, Literal, cast
from uuid import uuid4

from haiway import Meta, State, as_list, ctx
from ollama import ChatResponse, Image, Message, Options, Tool

from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputInvalid,
    ModelInstructions,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelOutputStream,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.models.metrics import record_model_invocation, record_usage_metrics
from draive.multimodal import MultimodalContent, TextContent
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
        **extra: Any,
    ) -> ModelOutputStream:
        return self._completion_stream(
            instructions=instructions,
            tools=tools,
            context=context,
            output=output,
            config=config or ctx.state(OllamaChatConfig),
            **extra,
        )

    async def _completion_stream(  # noqa: C901, PLR0912
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: OllamaChatConfig,
        **extra: Any,
    ) -> ModelOutputStream:
        async with ctx.scope("model.invocation"):
            record_model_invocation(
                provider="ollama",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=config.seed,
                thinking=config.thinking,
            )

            messages: list[Message]
            try:
                # eagerly materialize to convert context errors to ModelInputInvalid here
                messages = list(
                    _context_messages(
                        instructions=instructions,
                        context=context,
                    )
                )

            except Exception as exc:
                raise ModelInputInvalid(
                    provider="ollama",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            input_tokens: int | None = None
            output_tokens: int | None = None
            # the client declares a plain iterator while always producing an async
            # generator for streamed requests, the narrower type allows releasing it
            stream: AsyncGenerator[ChatResponse] | None = None
            try:
                stream = cast(
                    AsyncGenerator[ChatResponse],
                    await self._client.chat(  # pyright: ignore[reportUnknownMemberType]
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
                        think=unwrap_missing(config.thinking),
                        stream=True,
                    ),
                )

                async for chunk in stream:
                    # usage counters arrive within the terminal chunk
                    if chunk.prompt_eval_count is not None:
                        input_tokens = chunk.prompt_eval_count

                    if chunk.eval_count is not None:
                        output_tokens = chunk.eval_count

                    message: Message = chunk.message
                    if thinking := message.thinking:
                        yield ModelReasoningChunk.of(
                            TextContent.of(thinking),
                            meta={"kind": "thinking"},
                        )

                    # streaming chunks carry content fragments, images are never included
                    if content := message.content:
                        yield TextContent.of(content)

                    # ollama delivers each tool call complete within a single chunk
                    for request in _tool_calls_to_requests(message.tool_calls):
                        yield request

                    if chunk.done and chunk.done_reason == "length":
                        raise ModelOutputLimit(
                            provider="ollama",
                            model=config.model,
                            max_output_tokens=unwrap_missing(config.max_output_tokens) or 0,
                        )

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="ollama",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            finally:
                if input_tokens is not None or output_tokens is not None:
                    record_usage_metrics(
                        provider="ollama",
                        model=config.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                if stream is not None:
                    # release the http stream, iteration may have ended
                    # before the response was completed
                    await stream.aclose()


def _message_text(
    content: MultimodalContent,
    /,
) -> str:
    # images travel through a dedicated field, text resources have no representation
    # of their own - inlining keeps them within the message instead of dropping them
    return MultimodalContent.of(
        *(
            TextContent.of(part.to_bytes().decode())
            if isinstance(part, ResourceContent) and part.mime_type.startswith("text")
            else part
            for part in content.parts
            if not isinstance(part, ResourceReference | ResourceContent)
            or part.mime_type.startswith("text")
        )
    ).to_str()


def _content_images(
    content: MultimodalContent,
    /,
) -> list[Image] | None:
    images: list[Image] = []
    for resource in content.resources():
        mime_type: str = resource.mime_type or ""
        if mime_type.startswith("text"):
            continue  # inlined into the message text

        if not mime_type.startswith("image"):
            raise ValueError(f"Unsupported message content mime type: {mime_type}")

        if isinstance(resource, ResourceReference):
            # ollama accepts only raw base64 image data - neither urls nor data uris
            # the uri is omitted, it can carry credentials within its userinfo or query
            raise ValueError(
                f"Unsupported message content image reference ({resource.mime_type}),"
                " ollama accepts only inline image data"
            )

        # ResourceContent.data is already base64 encoded which is exactly what ollama expects
        images.append(Image(value=resource.data))

    return images or None


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
                    content=_message_text(content),
                    images=_content_images(content),
                )

            if responses := element.tool_responses:
                # Include any tool responses that follow the user message
                for tool_resp in responses:
                    yield Message(
                        role="tool",
                        tool_name=tool_resp.tool,
                        content=_message_text(tool_resp.content),
                    )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            content = element.content
            # reasoning travels through a dedicated field, dropping it would break
            # the thinking continuity across turns
            reasoning: str = "".join(
                block.reasoning.to_str()
                for block in element.output
                if isinstance(block, ModelReasoning)
            )
            yield Message(
                role="assistant",
                content=_message_text(content),
                thinking=reasoning or None,
                images=_content_images(content),
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
    # `Tool.Function.Parameters` is a closed model dropping everything but a handful of
    # keywords, erasing nested schemas - `model_construct` skips that lossy validation
    # and the client passes such an instance through `Tool.model_validate` unchanged
    return Tool.model_construct(
        type="function",
        function=Tool.Function.model_construct(
            name=tool.name,
            description=tool.description,
            parameters=cast(
                Any,
                tool.parameters
                or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ),
    )


def _tool_calls_to_requests(
    tool_calls: Sequence[Message.ToolCall] | None,
    /,
) -> list[ModelToolRequest]:
    if not tool_calls:
        return []

    # ollama reports no identifier and its tool messages carry only a tool name,
    # so results are correlated by name - repeating a tool within one response
    # leaves the model matching them positionally
    repeated: set[str] = {
        call.function.name
        for index, call in enumerate(tool_calls)
        if any(other.function.name == call.function.name for other in tool_calls[index + 1 :])
    }
    if repeated:
        ctx.log_warning(
            f"ollama requested {sorted(repeated)} more than once within a single response,"
            " their results can only be correlated by order"
        )

    return [
        ModelToolRequest(
            identifier=str(uuid4()),  # ollama does not return an id
            tool=call.function.name,
            # arguments are always delivered as an already decoded mapping
            arguments=call.function.arguments,
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
        ctx.log_warning(
            "ollama does not support required tool selection, using automatic selection instead",
        )
        return tools_list

    # specific tool suggestion is not supported by Ollama
    ctx.log_warning(
        f"ollama does not support selecting the {tool_selection.name} tool,"
        " using automatic selection instead",
    )
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
    disallowed_root_keys: set[str] = {"$schema", "$id", "$anchor"}
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

    if "$ref" in schema or any(key in schema for key in ("anyOf", "allOf", "not")):
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

    if output == "auto" or output == "text" or "text" in output:  # noqa: PLR1714
        return None

    # silently answering with text would not match the requested modality
    raise NotImplementedError(f"{output} output is not supported by Ollama")
