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
    Modality,
    Part,
    PartDict,
    SafetySettingDict,
    SchemaDict,
)
from haiway import Missing, as_dict

from draive.commons import META_EMPTY
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
    MediaData,
    MediaReference,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.multimodal.meta import MetaContent
from draive.parameters import DataModel

__all__ = (
    "DISABLED_SAFETY_SETTINGS",
    "content_element_as_part",
    "context_element_as_content",
    "output_as_response_declaration",
    "resolution_as_media_resolution",
    "result_part_as_content_or_call",
    "tools_as_tools_config",
)


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
                    "role": "user",
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
                                if response.handling == "error"
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


def content_element_as_part(  # noqa: PLR0911
    element: MultimodalContentElement,
    /,
) -> PartDict:
    match element:
        case TextContent() as text:
            return {
                "text": text.text,
            }

        case MediaData() as media_data:
            return {
                "inline_data": {
                    "data": media_data.data,
                    "mime_type": media_data.media,
                },
            }

        case MediaReference() as media_reference:
            return {
                "file_data": {
                    "file_uri": media_reference.uri,
                    "mime_type": media_reference.media,
                }
            }

        case MetaContent() as meta if meta.category == "thought":
            match meta.content:
                case None:
                    return {
                        "text": "",
                        "thought": True,
                    }

                case TextContent() as text:
                    return {
                        "text": text.text,
                        "thought": True,
                    }

                # not expecting media in thinking, treating it as json
                case DataModel() as model:
                    return {
                        "text": model.to_json(),
                        "thought": True,
                    }

        case DataModel() as data:
            return {
                "text": data.to_json(),
            }


def output_as_response_declaration(  # noqa: PLR0911
    output: LMMOutputSelection,
    /,
) -> tuple[
    SchemaDict | None,
    list[Modality] | None,
    str | None,
    Callable[[MultimodalContent], MultimodalContent],
]:
    match output:
        case "auto":
            # not specified at all - use defaults
            return (
                None,
                None,
                None,
                _auto_output_conversion,
            )

        case "text":
            return (
                None,
                [Modality.TEXT],
                "text/plain",
                _text_output_conversion,
            )

        case "json":
            return (
                None,
                [Modality.TEXT],
                "application/json",
                _json_output_conversion,
            )

        case "image":
            return (
                None,
                [Modality.TEXT, Modality.IMAGE],  # google api does not allow to specify only image
                None,  # define mime type?
                _image_output_conversion,  # we will ignore text anyways
            )

        case "audio":
            return (
                None,
                [Modality.AUDIO],
                None,  # define mime type?
                _audio_output_conversion,  # we will ignore text anyways
            )

        case "video":
            raise NotImplementedError("video output is not supported by Gemini")

        case ["text", "image"] | ["image", "text"]:  # refine multimodal matching?
            return (
                None,
                [Modality.TEXT, Modality.IMAGE],
                None,
                _auto_output_conversion,
            )

        case [*_]:
            raise NotImplementedError("multimodal output is not supported by Gemini")

        case model:
            return (
                cast(SchemaDict, model.__PARAMETERS_SPECIFICATION__),
                [Modality.TEXT],
                "application/json",
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


def _image_output_conversion(
    output: MultimodalContent,
    /,
) -> MultimodalContent:
    return MultimodalContent.of(*output.media("image"))


def _audio_output_conversion(
    output: MultimodalContent,
    /,
) -> MultimodalContent:
    return MultimodalContent.of(*output.media("audio"))


def _video_output_conversion(
    output: MultimodalContent,
    /,
) -> MultimodalContent:
    return MultimodalContent.of(*output.media("video"))


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


def result_part_as_content_or_call(
    part: Part,
    /,
) -> Iterable[MultimodalContentElement | LMMToolRequest]:
    result: list[MultimodalContentElement | LMMToolRequest] = []
    if part.text:
        # assuming only text thinking is possible
        if part.thought:
            result.append(
                MetaContent.of(
                    "thought",
                    content=TextContent(text=part.text),
                    meta=META_EMPTY,
                ),
            )

        else:
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
            MediaData.of(
                part.inline_data.data,
                media=part.inline_data.mime_type or "application/octet-stream",
            ),
        )

    if part.file_data and part.file_data.file_uri:  # there is no content without content...
        result.append(
            MediaReference.of(
                part.file_data.file_uri,
                media=part.file_data.mime_type or "application/octet-stream",
            ),
        )

    return result


def resolution_as_media_resolution(
    resolution: Literal["low", "medium", "high"] | Missing,
    /,
) -> MediaResolution | None:
    match resolution:
        case "low":
            return MediaResolution.MEDIA_RESOLUTION_LOW

        case "medium":
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
