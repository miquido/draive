from collections.abc import Generator, Sequence
from uuid import uuid4

from google.genai.types import (
    ContentDict,
    Part,
    PartDict,
)
from haiway import META_EMPTY, as_dict

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
    MetaContent,
    MultimodalContentElement,
    TextContent,
)

__all__ = (
    "context_element_as_content",
    "result_part_as_content_or_call",
)


def context_element_as_content(
    element: LMMContextElement,
    /,
) -> ContentDict:
    if isinstance(element, LMMInput):
        return {
            "role": "user",
            "parts": list(content_parts(element.content.parts)),
        }

    elif isinstance(element, LMMCompletion):
        return {
            "role": "model",
            "parts": list(content_parts(element.content.parts)),
        }

    elif isinstance(element, LMMToolRequests):
        return {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "id": request.identifier,
                        "name": request.tool,
                        "args": as_dict(request.arguments),
                    }
                }
                for request in element.requests
            ],
        }

    elif isinstance(element, LMMToolResponses):
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": response.identifier,
                        "name": response.tool,
                        "response": {
                            "error": list(content_parts(response.content.parts)),
                        }
                        if response.handling == "error"
                        else {
                            "output": list(content_parts(response.content.parts)),
                        },
                    }
                }
                for response in element.responses
            ],
        }


def content_parts(
    elements: Sequence[MultimodalContentElement],
    /,
) -> Generator[PartDict]:
    for element in elements:
        if isinstance(element, TextContent):
            yield {
                "text": element.text,
            }

        elif isinstance(element, MediaData):
            yield {
                "inline_data": {
                    "data": element.data,
                    "mime_type": element.media,
                },
            }

        elif isinstance(element, MediaReference):
            yield {
                "file_data": {
                    "file_uri": element.uri,
                    "mime_type": element.media,
                }
            }

        elif isinstance(element, MetaContent):
            if element.category == "thought" and element.content:
                yield {
                    "text": element.content.to_str(),
                    "thought": True,
                }

            elif element.category == "transcript" and element.content:
                yield {
                    "text": element.content.to_str(),
                }

            else:
                continue  # skip other meta

        else:
            yield {
                "text": element.to_json(),
            }


def result_part_as_content_or_call(
    part: Part,
    /,
) -> Generator[MultimodalContentElement | LMMToolRequest]:
    if part.text:
        # assuming only text thinking is possible
        if part.thought:
            yield MetaContent.of(
                "thought",
                content=TextContent.of(part.text),
                meta=META_EMPTY,
            )

        else:
            yield TextContent.of(part.text)

    if part.function_call and part.function_call.name:  # can't call without a name
        yield LMMToolRequest(
            identifier=part.function_call.id or uuid4().hex,
            tool=part.function_call.name,
            arguments=part.function_call.args or {},
        )

    if part.inline_data and part.inline_data.data:  # there is no content without content...
        yield MediaData.of(
            part.inline_data.data,
            media=part.inline_data.mime_type or "application/octet-stream",
        )

    if part.file_data and part.file_data.file_uri:  # there is no content without content...
        yield MediaReference.of(
            part.file_data.file_uri,
            media=part.file_data.mime_type or "application/octet-stream",
        )
