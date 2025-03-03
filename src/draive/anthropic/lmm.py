from base64 import b64encode
from collections.abc import Callable, Iterable
from typing import Any, cast

from anthropic import NOT_GIVEN, NotGiven
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolParam,
)
from haiway import Missing

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
    MediaContent,
    Multimodal,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.parameters import DataModel

__all__ = [
    "content_block_as_content_element",
    "context_element_as_message",
    "convert_content_element",
    "thinking_budget_as_config",
    "tools_as_tool_config",
]


def context_element_as_message(
    element: LMMContextElement,
) -> MessageParam:
    match element:
        case LMMInput() as input:
            return {
                "role": "user",
                "content": [
                    convert_content_element(element=element) for element in input.content.parts
                ],
            }

        case LMMCompletion() as completion:
            return {
                "role": "assistant",
                "content": [
                    convert_content_element(element=element) for element in completion.content.parts
                ],
            }

        case LMMToolRequests() as tool_requests:
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
                        "is_error": response.error,
                        "content": [
                            convert_content_element(element=part) for part in response.content.parts
                        ],
                    }
                    for response in tool_responses.responses
                ],
            }


def convert_content_element(
    element: MultimodalContentElement,
) -> TextBlockParam | ImageBlockParam:
    match element:
        case TextContent() as text:
            return {
                "type": "text",
                "text": text.text,
            }

        case MediaContent() as media:
            if media.kind != "image":
                raise ValueError("Unsupported message content", media)

            match media.source:
                case str() as url:
                    return {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        },
                    }

                case bytes() as data:
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": cast(Any, media.media),
                            "data": b64encode(data).decode(),
                        },
                    }

        case DataModel() as data:
            return {
                "type": "text",
                "text": data.as_json(),
            }


def content_block_as_content_element(
    block: TextBlock,
    /,
) -> MultimodalContentElement:
    match block:
        case TextBlock() as text:
            return TextContent(text=text.text)


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
) -> tuple[MultimodalContent | None, Callable[[MultimodalContent], Multimodal]]:
    match output:
        case "auto":
            return (None, _auto_output_conversion)

        case "text":
            return (None, _text_output_conversion)

        case "json":
            return (MultimodalContent.of("{"), _json_output_conversion)

        case "image":
            raise NotImplementedError("image output is not supported by Anthropic")

        case "audio":
            raise NotImplementedError("audio output is not supported by Anthropic")

        case "video":
            raise NotImplementedError("video output is not supported by Anthropic")

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
) -> Multimodal:
    return output


def _text_output_conversion(
    output: MultimodalContent,
    /,
) -> Multimodal:
    return output.as_string()


def _json_output_conversion(
    output: MultimodalContent,
    /,
) -> Multimodal:
    return MultimodalContent.of(DataModel.from_json(output.as_string()))


def _prepare_model_output_conversion(
    model: type[DataModel],
    /,
) -> Callable[[MultimodalContent], Multimodal]:
    def _model_output_conversion(
        output: MultimodalContent,
        /,
    ) -> Multimodal:
        return MultimodalContent.of(model.from_json(output.as_string()))

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
