import json
from base64 import b64encode
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from haiway import not_missing
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionModality,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema

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
from draive.openai.config import OpenAIChatConfig
from draive.parameters import DataModel

__all__ = [
    "content_element_as_content_part",
    "context_element_as_messages",
    "tools_as_tool_config",
]


def content_element_as_content_part(
    element: MultimodalContentElement,
    config: OpenAIChatConfig,
) -> ChatCompletionContentPartParam:
    match element:
        case TextContent() as text:
            return {
                "type": "text",
                "text": text.text,
            }

        case MediaContent() as media:
            if media.kind != "image":
                raise ValueError("Unsupported message content", media)

            url: str
            match media.source:
                case str() as string:
                    url = string

                case bytes() as data:
                    url = f"data:{media.media};base64,{b64encode(data).decode()}"

            return {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": cast(Literal["auto", "low", "high"], config.vision_details)
                    if not_missing(config.vision_details)
                    else "auto",
                },
            }

        case DataModel() as data:
            return {
                "type": "text",
                "text": data.as_json(),
            }


def context_element_as_messages(
    element: LMMContextElement,
    /,
    config: OpenAIChatConfig,
) -> Iterable[ChatCompletionMessageParam]:
    match element:
        case LMMInput() as input:
            return (
                {
                    "role": "user",
                    "content": [
                        content_element_as_content_part(
                            element=element,
                            config=config,
                        )
                        for element in input.content.parts
                    ],
                },
            )

        case LMMCompletion() as completion:
            return (
                {
                    "role": "assistant",
                    # TODO: OpenAI models generating media?
                    "content": completion.content.as_string(),
                },
            )

        case LMMToolRequests() as tool_requests:
            return (
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": request.identifier,
                            "type": "function",
                            "function": {
                                "name": request.tool,
                                "arguments": json.dumps(dict(request.arguments)),
                            },
                        }
                        for request in tool_requests.requests
                    ],
                },
            )

        case LMMToolResponses() as tool_responses:
            return (
                {
                    "role": "tool",
                    "tool_call_id": response.identifier,
                    "content": response.content.as_string(),
                }
                for response in tool_responses.responses
            )


def output_as_response_declaration(  # noqa: PLR0911
    output: LMMOutputSelection,
) -> tuple[
    ResponseFormat | ResponseFormatJSONSchema | NotGiven,
    list[ChatCompletionModality] | NotGiven,
    Callable[[MultimodalContent], Multimodal],
]:
    match output:
        case "auto":
            return (
                NOT_GIVEN,
                NOT_GIVEN,
                _auto_output_conversion,
            )

        case ["text"] | "text":
            return (
                {"type": "text"},
                ["text"],
                _auto_output_conversion,
            )

        case "json":
            return (
                {"type": "json_object"},
                ["text"],
                _auto_output_conversion,
            )
            return ({"type": "json_object"}, _json_output_conversion)

        case "image":
            raise NotImplementedError("image output is not supported by OpenAI")

        case ["audio"] | "audio":
            return (NOT_GIVEN, ["audio"], _audio_output_conversion)

        case "video":
            raise NotImplementedError("video output is not supported by OpenAI")

        case ["text", "audio"] | ["audio", "text"]:  # refine multimodal matching?
            return (NOT_GIVEN, ["text", "audio"], _auto_output_conversion)

        case [*_]:
            raise NotImplementedError("multimodal output is not supported by OpenAI")

        case model:
            return (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": model.__name__,
                        "schema": cast(
                            dict[str, Any],
                            model.__PARAMETERS_SPECIFICATION__,
                        ),
                        "strict": False,
                    },
                },
                ["text"],
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


def _audio_output_conversion(
    output: MultimodalContent,
    /,
) -> Multimodal:
    return MultimodalContent.of(*output.media("audio"))


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


def tool_specification_as_tool(
    tool: LMMToolSpecification,
    /,
) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"] or "",
            "parameters": cast(dict[str, Any], tool["parameters"])
            or {"type": "object", "properties": {}},
        },
    }


def tools_as_tool_config(
    tools: Iterable[LMMToolSpecification] | None,
    /,
    *,
    tool_selection: LMMToolSelection,
) -> tuple[
    ChatCompletionToolChoiceOptionParam | NotGiven, list[ChatCompletionToolParam] | NotGiven
]:
    tools_list: list[ChatCompletionToolParam] = [
        tool_specification_as_tool(tool) for tool in (tools or [])
    ]
    if not tools_list:
        return (NOT_GIVEN, NOT_GIVEN)

    match tool_selection:
        case "auto":
            return ("auto", tools_list)

        case "none":
            return (NOT_GIVEN, NOT_GIVEN)

        case "required":
            return ("required", tools_list)

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101

            return (
                {
                    "type": "function",
                    "function": {"name": tool["name"]},
                },
                tools_list,
            )
