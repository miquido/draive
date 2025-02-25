from collections.abc import Callable, Iterable
from typing import Literal, cast
from uuid import uuid4

from google.genai.types import (
    ContentDict,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Part,
    PartDict,
    SafetySettingDict,
    SchemaDict,
)
from haiway import Missing, as_dict

from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMOutputSelection,
    LMMToolRequest,
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
    "DISABLED_SAFETY_SETTINGS",
    "content_element_as_part",
    "context_element_as_content",
    "output_as_response_declaration",
    "resoluton_as_media_resulution",
    "result_part_as_content_or_call",
    "tools_as_tools_config",
]


def context_element_as_content(
    element: LMMContextElement,
    /,
) -> Iterable[ContentDict]:
    match element:
        case LMMInput() as input:
            return (
                {
                    "role": "user",
                    "parts": [content_element_as_part(element) for element in input.content.parts],
                },
            )

        case LMMCompletion() as completion:
            return (
                {
                    "role": "model",
                    "parts": [
                        content_element_as_part(element) for element in completion.content.parts
                    ],
                },
            )

        case LMMToolRequests() as tool_requests:
            return (
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "id": request.identifier,
                                "name": request.tool,
                                "args": as_dict(request.arguments),
                            }
                        }
                        for request in tool_requests.requests
                    ],
                },
            )

        case LMMToolResponses() as tool_responses:
            return (
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_response": {
                                "id": response.identifier,
                                "name": response.tool,
                                "response": {
                                    "error": [
                                        content_element_as_part(element)
                                        for element in response.content.parts
                                    ],
                                }
                                if response.error
                                else {
                                    "output": [
                                        content_element_as_part(element)
                                        for element in response.content.parts
                                    ],
                                },
                            }
                        }
                        for response in tool_responses.responses
                    ],
                },
            )


def content_element_as_part(
    element: MultimodalContentElement,
    /,
) -> PartDict:
    match element:
        case TextContent() as text:
            return {
                "text": text.text,
            }

        case MediaContent() as media:
            match media.source:
                case str() as uri:
                    return {
                        "file_data": {
                            "file_uri": uri,
                            "mime_type": media.media,
                        }
                    }

                case bytes() as data:
                    return {
                        "inline_data": {
                            "data": data,
                            "mime_type": media.media,
                        },
                    }

        case DataModel() as data:
            return {
                "text": data.as_json(),
            }


def output_as_response_declaration(  # noqa: PLR0911
    output: LMMOutputSelection,
    /,
) -> tuple[SchemaDict | None, str | None, Callable[[MultimodalContent], Multimodal]]:
    match output:
        case "auto":
            return (None, None, _auto_output_conversion)

        case "text":
            return (None, "text/plain", _text_output_conversion)

        case "json":
            return (
                None,
                "application/json",
                _json_output_conversion,
            )

        case "image":
            return (
                None,
                "image/png",
                _image_output_conversion,
            )  # TODO: verify with actual capabilities

        case "audio":
            return (
                None,
                "audio/wav",
                _audio_output_conversion,
            )  # TODO: verify with actual capabilities

        case "video":
            return (
                None,
                "video/mpeg",
                _video_output_conversion,
            )  # TODO: verify with actual capabilities

        case model:
            return (
                cast(SchemaDict, model.__PARAMETERS_SPECIFICATION__),
                "application/json",
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


def _image_output_conversion(
    output: MultimodalContent,
    /,
) -> Multimodal:
    return MultimodalContent.of(*output.media("image"))


def _audio_output_conversion(
    output: MultimodalContent,
    /,
) -> Multimodal:
    return MultimodalContent.of(*output.media("audio"))


def _video_output_conversion(
    output: MultimodalContent,
    /,
) -> Multimodal:
    return MultimodalContent.of(*output.media("video"))


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


def result_part_as_content_or_call(
    part: Part,
    /,
) -> Iterable[MultimodalContentElement | LMMToolRequest]:
    result: list[MultimodalContentElement | LMMToolRequest] = []
    if part.text:
        result.append(TextContent(text=part.text))

    if part.function_call and part.function_call.name:  # can't call without a name
        result.append(
            LMMToolRequest(
                identifier=part.function_call.id or uuid4().hex,
                tool=part.function_call.name,
                arguments=part.function_call.args or {},
            )
        )

    if part.inline_data and part.inline_data.data:  # there is no content without content...
        result.append(
            MediaContent.data(
                part.inline_data.data,
                media=part.inline_data.mime_type or "unknown",
            ),
        )

    if part.file_data and part.file_data.file_uri:  # there is no content without content...
        result.append(
            MediaContent.url(
                part.file_data.file_uri,
                media=part.file_data.mime_type or "unknown",
            ),
        )

    # TODO: we could also extract thoughts
    return result


def resoluton_as_media_resulution(
    resolution: Literal["low", "medium", "high"] | Missing,
    /,
) -> MediaResolution | None:
    match resolution:
        case "low":
            return MediaResolution.MEDIA_RESOLUTION_LOW

        case "modeium":
            return MediaResolution.MEDIA_RESOLUTION_MEDIUM

        case "high":
            return MediaResolution.MEDIA_RESOLUTION_HIGH

        case _:
            return None


def tools_as_tools_config(
    tools: Iterable[LMMToolSpecification] | None,
    /,
    tool_selection: LMMToolSelection,
) -> tuple[list[FunctionDeclarationDict] | None, FunctionCallingConfigMode | None]:
    functions: list[FunctionDeclarationDict] = []
    for tool in tools or []:
        declaration: FunctionDeclarationDict = FunctionDeclarationDict(
            name=tool["name"],
            description=tool["description"],
            parameters=cast(SchemaDict, tool["parameters"]),
        )

        functions.append(declaration)

    if not functions:
        return (
            None,
            FunctionCallingConfigMode.NONE,
        )

    match tool_selection:
        case "auto":
            return (
                functions,
                FunctionCallingConfigMode.AUTO,
            )

        case "required":
            return (
                functions,
                FunctionCallingConfigMode.ANY,
            )

        case "none":
            return (
                None,  # no need to pass functions if none can be used
                FunctionCallingConfigMode.NONE,
            )

        case _:  # TODO: FIXME: specific tool selection?
            return (
                functions,
                FunctionCallingConfigMode.AUTO,
            )


DISABLED_SAFETY_SETTINGS: list[SafetySettingDict] = [
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.OFF,
    ),
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.OFF,
    ),
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.OFF,
    ),
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.OFF,
    ),
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold=HarmBlockThreshold.OFF,
    ),
]
