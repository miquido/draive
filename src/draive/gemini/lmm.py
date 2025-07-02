from collections.abc import Iterable
from uuid import uuid4

from google.genai.types import (
    ContentDict,
    Part,
    PartDict,
)
from haiway import as_dict

from draive.commons import META_EMPTY
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponses,
)
from draive.multimodal import (
    MediaData,
    MediaReference,
    MultimodalContentElement,
    TextContent,
)
from draive.multimodal.meta import MetaContent
from draive.parameters import DataModel

__all__ = (
    "content_element_as_part",
    "context_element_as_content",
    "result_part_as_content_or_call",
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
                    content=TextContent.of(part.text),
                    meta=META_EMPTY,
                ),
            )

        else:
            result.append(TextContent.of(part.text))

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
