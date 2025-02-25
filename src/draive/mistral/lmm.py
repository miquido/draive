import json
from base64 import b64encode
from collections.abc import Callable, Iterable
from typing import Any, cast

from mistralai.models import (
    ChatCompletionRequestToolChoiceTypedDict,
    ContentChunk,
    ContentChunkTypedDict,
    ImageURLChunk,
    MessagesTypedDict,
    ResponseFormatTypedDict,
    TextChunk,
    ToolTypedDict,
)

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
from draive.mistral.types import MistralException
from draive.multimodal import (
    MediaContent,
    Multimodal,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.parameters import DataModel

__all__ = [
    "content_chunk_as_content_element",
    "content_element_as_content_chunk",
    "context_element_as_messages",
    "output_as_response_declaration",
    "tool_specification_as_tool",
    "tools_as_tool_config",
]


def context_element_as_messages(
    element: LMMContextElement,
    /,
) -> Iterable[MessagesTypedDict]:
    match element:
        case LMMInput() as input:
            return (
                {
                    "role": "user",
                    "content": [
                        content_element_as_content_chunk(element) for element in input.content.parts
                    ],
                },
            )

        case LMMCompletion() as completion:
            return (
                {
                    "role": "assistant",
                    "content": [
                        content_element_as_content_chunk(element)
                        for element in completion.content.parts
                    ],
                },
            )

        case LMMToolRequests() as tool_requests:
            return (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": request.identifier,
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
                    "name": response.tool,
                    "content": [
                        content_element_as_content_chunk(element)
                        for element in response.content.parts
                    ],
                }
                for response in tool_responses.responses
            )


def content_element_as_content_chunk(
    element: MultimodalContentElement,
    /,
) -> ContentChunkTypedDict:
    match element:
        case TextContent() as text:
            return {
                "type": "text",
                "text": text.text,
            }

        case MediaContent() as media:
            if media.kind != "image":
                raise ValueError("Unsupported content", media)

            url: str
            match media.source:
                case str() as uri:
                    url = uri

                case bytes() as data:
                    url = f"data:{media.media};base64,{b64encode(data).decode()}"

            return {
                "type": "image_url",
                "image_url": {"url": url},
                # TODO: there is optional "detail" argument, however undocumented
            }

        case DataModel() as data:
            return {
                "type": "text",
                "text": data.as_json(),
            }


def content_chunk_as_content_element(
    element: ContentChunk,
    /,
) -> MultimodalContentElement:
    match element:
        case TextChunk() as chunk:
            return TextContent(text=chunk.text)

        case ImageURLChunk() as image:
            match image.image_url:
                case str() as url:
                    return MediaContent.url(
                        url,
                        media="image",
                    )

                case image_url:
                    return MediaContent.url(
                        image_url.url,
                        media="image",
                    )

        case other:
            raise MistralException("Unsupported Mistral message content chunk", other)


def tool_specification_as_tool(
    tool: LMMToolSpecification,
    /,
) -> ToolTypedDict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"] or "",
            "parameters": cast(dict[str, Any], tool["parameters"]),
        },
    }


def output_as_response_declaration(
    output: LMMOutputSelection,
) -> tuple[ResponseFormatTypedDict, Callable[[MultimodalContent], Multimodal]]:
    match output:
        case "auto":
            return ({"type": "text"}, _auto_output_conversion)

        case "text":
            return ({"type": "text"}, _text_output_conversion)

        case "json":
            return ({"type": "json_object"}, _json_output_conversion)

        case "image":
            raise NotImplementedError("image output is not supported by Mistral")

        case "audio":
            raise NotImplementedError("audio output is not supported by Mistral")

        case "video":
            raise NotImplementedError("video output is not supported by Mistral")

        case model:
            return (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": model.__name__,
                        "schema_definition": cast(
                            dict[str, Any],
                            model.__PARAMETERS_SPECIFICATION__,
                        ),
                        "description": None,
                        "strict": False,
                    },
                },
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
        return MultimodalContent.of(DataModel.from_json(output.as_string()))

    return _model_output_conversion


def tools_as_tool_config(
    tools: Iterable[LMMToolSpecification] | None,
    /,
    *,
    tool_selection: LMMToolSelection,
) -> tuple[ChatCompletionRequestToolChoiceTypedDict, list[ToolTypedDict]]:
    tools_list: list[ToolTypedDict] = [tool_specification_as_tool(tool) for tool in (tools or [])]
    if not tools_list:
        return ("none", tools_list)

    match tool_selection:
        case "auto":
            return ("auto", tools_list)

        case "none":
            return ("none", [])

        case "required":
            return ("any", tools_list)

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101

            return (
                {
                    "type": "function",
                    "function": {"name": tool["name"]},
                },
                tools_list,
            )
