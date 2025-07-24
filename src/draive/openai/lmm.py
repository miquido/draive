import json
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, Literal, cast, overload

from haiway import Missing, not_missing
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
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
    MediaData,
    MediaReference,
    MetaContent,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.parameters import DataModel

__all__ = (
    "content_parts",
    "context_element_as_messages",
    "tools_as_tool_config",
)


@overload
def content_parts(
    content: Sequence[MultimodalContentElement],
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Missing,
    text_only: Literal[True],
) -> Generator[ChatCompletionContentPartTextParam]: ...


@overload
def content_parts(
    content: Sequence[MultimodalContentElement],
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Missing,
    text_only: Literal[False],
) -> Generator[ChatCompletionContentPartParam]: ...


def content_parts(  # noqa: C901, PLR0912
    content: Sequence[MultimodalContentElement],
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Missing,
    text_only: bool,
) -> Generator[ChatCompletionContentPartParam]:
    for element in content:
        if isinstance(element, TextContent):
            yield {
                "type": "text",
                "text": element.text,
            }

        elif isinstance(element, MediaData):
            if element.kind != "image":
                raise ValueError("Unsupported message content", element)

            if text_only:
                yield {
                    "type": "text",
                    "text": element.to_str(),
                }

            else:
                yield {
                    "type": "image_url",
                    "image_url": {
                        "url": element.to_data_uri(safe_encoding=False),
                        "detail": cast(Literal["auto", "low", "high"], vision_details)
                        if not_missing(vision_details)
                        else "auto",
                    },
                }

        elif isinstance(element, MediaReference):
            if element.kind != "image":
                raise ValueError("Unsupported message content", element)

            if text_only:
                yield {
                    "type": "text",
                    "text": element.to_str(),
                }

            else:
                yield {
                    "type": "image_url",
                    "image_url": {
                        "url": element.uri,
                        "detail": cast(Literal["auto", "low", "high"], vision_details)
                        if not_missing(vision_details)
                        else "auto",
                    },
                }

        elif isinstance(element, MetaContent):
            if element.category == "transcript" and element.content:
                yield {
                    "type": "text",
                    "text": element.content.to_str(),
                }

            else:
                continue  # skip other meta

        else:  # DataModel
            yield {
                "type": "text",
                "text": element.to_json(),
            }


def context_element_as_messages(
    element: LMMContextElement,
    /,
    vision_details: Literal["auto", "low", "high"] | Missing,
) -> Iterable[ChatCompletionMessageParam]:
    match element:
        case LMMInput() as input:
            return (
                {
                    "role": "user",
                    "content": list(
                        content_parts(
                            input.content.parts,
                            vision_details=vision_details,
                            text_only=False,
                        )
                    ),
                },
            )

        case LMMCompletion() as completion:
            return (
                {
                    "role": "assistant",
                    "content": list(
                        content_parts(
                            completion.content.parts,
                            vision_details=vision_details,
                            text_only=True,
                        )
                    ),
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
                    "content": list(
                        content_parts(
                            response.content.parts,
                            vision_details=vision_details,
                            text_only=True,
                        )
                    ),
                }
                for response in tool_responses.responses
            )


def output_as_response_declaration(
    output: LMMOutputSelection,
) -> tuple[
    ResponseFormat | ResponseFormatJSONSchema | NotGiven,
    Callable[[MultimodalContent], MultimodalContent],
]:
    match output:
        case "auto":
            return (
                NOT_GIVEN,
                _auto_output_conversion,
            )

        case ["text"] | "text":
            return (
                {"type": "text"},
                _auto_output_conversion,
            )

        case "json":
            return (
                {"type": "json_object"},
                _json_output_conversion,
            )

        case "image":
            raise NotImplementedError("image output is not supported yet")

        case "audio":
            raise NotImplementedError("audio output is not supported yet")

        case "video":
            raise NotImplementedError("video output is not supported yet")

        case [*_]:
            raise NotImplementedError("multimodal output is not supported yet")

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


def _audio_output_conversion(
    output: MultimodalContent,
    /,
) -> MultimodalContent:
    return MultimodalContent.of(*output.media("audio"))


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
