from base64 import b64encode
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, cast

from anthropic import NOT_GIVEN, NotGiven
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    RedactedThinkingBlock,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolParam,
)
from anthropic.types.redacted_thinking_block_param import RedactedThinkingBlockParam
from anthropic.types.thinking_block_param import ThinkingBlockParam
from haiway import META_EMPTY, Missing

from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMOutputSelection,
    LMMToolRequests,
    LMMToolResponses,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.multimodal import (
    MediaData,
    MediaReference,
    MetaContent,
    Multimodal,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.parameters import DataModel

__all__ = (
    "content_block_as_content_element",
    "content_elements",
    "context_element_as_message",
    "thinking_budget_as_config",
    "tools_as_tool_config",
)


def context_element_as_message(  # noqa: C901
    element: LMMContextElement,
) -> MessageParam:
    match element:
        case LMMInput() as input:
            message_content: list[
                TextBlockParam | ImageBlockParam | ThinkingBlockParam | RedactedThinkingBlockParam
            ] = list(content_elements(input.content.parts))

            if input.meta.get("cache") == "ephemeral":
                for message in reversed(message_content):
                    if message["type"] in ("text", "image"):
                        message["cache_control"] = {  # pyright: ignore[reportGeneralTypeIssues]
                            "type": "ephemeral",
                        }
                        break

            return {
                "role": "user",
                "content": message_content,
            }

        case LMMCompletion() as completion:
            message_content: list[
                TextBlockParam | ImageBlockParam | ThinkingBlockParam | RedactedThinkingBlockParam
            ] = list(content_elements(completion.content.parts))

            if completion.meta.get("cache") == "ephemeral":
                for message in reversed(message_content):
                    if message["type"] in ("text", "image"):
                        message["cache_control"] = {  # pyright: ignore[reportGeneralTypeIssues]
                            "type": "ephemeral",
                        }
                        break

            return {
                "role": "assistant",
                "content": message_content,
            }

        case LMMToolRequests() as tool_requests:
            if tool_requests.content:
                return {
                    "role": "assistant",
                    "content": [
                        *content_elements(tool_requests.content.parts),
                        *[
                            {
                                "id": request.identifier,
                                "type": "tool_use",
                                "name": request.tool,
                                "input": request.arguments,
                            }
                            for request in tool_requests.requests
                        ],
                    ],
                }

            else:
                return {
                    "role": "assistant",
                    "content": [
                        {
                            "id": request.identifier,
                            "type": "tool_use",
                            "name": request.tool,
                            "input": request.arguments,
                        }
                        for request in tool_requests.requests
                    ],
                }

        case LMMToolResponses() as tool_responses:
            return {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": response.identifier,
                        "type": "tool_result",
                        "is_error": response.handling == "error",
                        "content": cast(  # there will be no thinking within tool results
                            Iterable[TextBlockParam | ImageBlockParam],
                            content_elements(response.content.parts),
                        ),
                    }
                    for response in tool_responses.responses
                ],
            }


def content_elements(  # noqa: C901
    elements: Sequence[MultimodalContentElement],
) -> Generator[TextBlockParam | ImageBlockParam | ThinkingBlockParam | RedactedThinkingBlockParam]:
    for element in elements:
        if isinstance(element, TextContent):
            yield {
                "type": "text",
                "text": element.text,
            }

        elif isinstance(element, MediaData):
            if element.kind != "image":
                raise ValueError("Unsupported message content", element)

            yield {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": cast(Any, element.media),
                    "data": b64encode(element.data).decode(),
                },
            }

        elif isinstance(element, MediaReference):
            if element.kind != "image":
                raise ValueError("Unsupported message content", element)

            yield {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": element.uri,
                },
            }

        elif isinstance(element, MetaContent):
            if element.category == "thinking":
                yield {
                    "type": "thinking",
                    "thinking": element.content.to_str() if element.content else "",
                    "signature": element.meta.get_str("signature", default=""),
                }

            elif element.category == "redacted_thinking":
                yield {
                    "type": "redacted_thinking",
                    "data": element.content.to_str() if element.content else "",
                }

            elif element.category == "transcript" and element.content:
                yield {
                    "type": "text",
                    "text": element.content.to_str(),
                }

            else:
                continue  # skip other meta

        else:
            yield {
                "type": "text",
                "text": element.to_json(),
            }


def content_block_as_content_element(
    block: TextBlock | ThinkingBlock | RedactedThinkingBlock,
    /,
) -> MultimodalContentElement:
    match block:
        case TextBlock() as text:
            return TextContent(
                text=text.text,
                # TODO: add citations meta
            )

        case ThinkingBlock() as thinking:
            return MetaContent.of(
                "thinking",
                content=TextContent(
                    text=thinking.thinking,
                ),
                meta={
                    "signature": thinking.signature,
                },
            )

        case RedactedThinkingBlock() as redacted_thinking:
            return MetaContent.of(
                "redacted_thinking",
                content=TextContent(
                    text=redacted_thinking.data,
                ),
                meta=META_EMPTY,
            )


def thinking_budget_as_config(budget: int | Missing) -> ThinkingConfigParam:
    match budget:
        case int() as budget:
            return {
                "type": "enabled",
                "budget_tokens": budget,
            }

        case _:
            return {"type": "disabled"}


def tool_specification_as_tool_param(
    tool: LMMToolSpecification,
    /,
) -> ToolParam:
    return {
        "name": tool["name"],
        "description": tool["description"] or "",
        "input_schema": cast(dict[str, Any], tool["parameters"]),
    }


def output_as_response_declaration(
    *,
    output: LMMOutputSelection,
    prefill: Multimodal | None,
) -> tuple[MultimodalContent | None, Callable[[MultimodalContent], MultimodalContent]]:
    match output:
        case "auto":
            return (None, _auto_output_conversion)

        case ["text"] | "text":
            return (None, _text_output_conversion)

        case "json":
            return (MultimodalContent.of("{"), _json_output_conversion)

        case "image":
            raise NotImplementedError("image output is not supported by Anthropic")

        case "audio":
            raise NotImplementedError("audio output is not supported by Anthropic")

        case "video":
            raise NotImplementedError("video output is not supported by Anthropic")

        case [*_]:
            raise NotImplementedError("multimodal output is not supported by Anthropic")

        case model:
            # we can't really do much better in this case
            # output conversion would require prompt changes to succeed
            # as we are not using the schema in this case
            # although we could use thinking tokens prefill if able
            # to inject the schema somehow unless that would be en error
            return (
                MultimodalContent.of("{"),
                _prepare_model_output_conversion(model),
            )


def _auto_output_conversion(
    output: MultimodalContent,
    /,
) -> MultimodalContent:
    return output


def _text_output_conversion(
    output: MultimodalContent,
    /,
) -> MultimodalContent:
    return MultimodalContent.of(output.to_str())


def _json_output_conversion(
    output: MultimodalContent,
    /,
) -> MultimodalContent:
    return MultimodalContent.of(DataModel.from_json(output.to_str()))


def _prepare_model_output_conversion(
    model: type[DataModel],
    /,
) -> Callable[[MultimodalContent], MultimodalContent]:
    def _model_output_conversion(
        output: MultimodalContent,
        /,
    ) -> MultimodalContent:
        return MultimodalContent.of(model.from_json(output.to_str()))

    return _model_output_conversion


def tools_as_tool_config(
    tools: Iterable[LMMToolSpecification] | None,
    /,
    *,
    tool_selection: LMMToolSelection,
) -> tuple[ToolChoiceParam | NotGiven, list[ToolParam] | NotGiven]:
    tools_list: list[ToolParam] = [tool_specification_as_tool_param(tool) for tool in (tools or [])]
    if not tools_list:
        return (NOT_GIVEN, NOT_GIVEN)

    match tool_selection:
        case "auto":
            return ({"type": "auto"}, tools_list)

        case "none":
            return (NOT_GIVEN, NOT_GIVEN)

        case "required":
            return ({"type": "any"}, tools_list)

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101

            return (
                {
                    "type": "tool",
                    "name": tool["name"],
                },
                tools_list,
            )
