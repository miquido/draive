from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, Literal, cast

from draive.bedrock.models import ChatMessage, ChatMessageContent, ChatTool
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponses,
    LMMToolSpecification,
)
from draive.lmm.types import LMMOutputSelection, LMMToolSelection
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
    "convert_context_element",
    "convert_tool",
)


def convert_content(  # noqa: C901, PLR0912
    elements: Sequence[MultimodalContentElement],
) -> Generator[ChatMessageContent]:
    for element in elements:
        if isinstance(element, TextContent):
            yield {"text": element.text}

        elif isinstance(element, MediaData):
            if element.kind != "image":
                raise ValueError("Unsupported message content", element)

            image_format: Literal["png", "jpeg", "gif"]
            match element.media:
                case "image/png":
                    image_format = "png"

                case "image/jpeg":
                    image_format = "jpeg"

                case "image/gif":
                    image_format = "gif"

                case _:
                    raise ValueError("Unsupported message content", element)

            yield {
                "image": {
                    "format": image_format,
                    "source": {"bytes": element.data},
                }
            }

        elif isinstance(element, MediaReference):
            raise ValueError("Unsupported message content", element)

        elif isinstance(element, MetaContent):
            if element.category == "transcript" and element.content:
                yield {
                    "text": element.content.to_str(),
                }

            else:
                continue  # skip other meta

        else:
            yield {"text": element.to_json()}


def convert_context_element(
    element: LMMContextElement,
) -> ChatMessage:
    match element:
        case LMMInput() as input:
            return ChatMessage(
                role="user",
                content=list(convert_content(input.content.parts)),
            )

        case LMMCompletion() as completion:
            return ChatMessage(
                role="assistant",
                content=list(convert_content(completion.content.parts)),
            )

        case LMMToolRequests() as requests:
            return ChatMessage(
                role="assistant",
                content=[
                    {
                        "toolUse": {
                            "toolUseId": request.identifier,
                            "name": request.tool,
                            "input": request.arguments,
                        }
                    }
                    for request in requests.requests
                ],
            )

        case LMMToolResponses() as tool_responses:
            return ChatMessage(
                role="user",
                content=[
                    {
                        "toolResult": {
                            "toolUseId": response.identifier,
                            "content": cast(Any, list(convert_content(response.content.parts))),
                            "status": "error" if response.handling == "error" else "success",
                        },
                    }
                    for response in tool_responses.responses
                ],
            )


def convert_tool(tool: LMMToolSpecification) -> ChatTool:
    return {
        "name": tool["name"],
        "description": tool["description"] or "",
        "inputSchema": {"json": tool["parameters"]},
    }


def output_as_response_declaration(
    output: LMMOutputSelection,
) -> Callable[[MultimodalContent], MultimodalContent]:
    match output:
        case "auto":
            return _auto_output_conversion

        case ["text"] | "text":
            return _auto_output_conversion

        case "json":
            return _json_output_conversion

        case "image":
            raise NotImplementedError("image output is not supported yet")

        case "audio":
            raise NotImplementedError("audio output is not supported yet")

        case "video":
            raise NotImplementedError("video output is not supported yet")

        case [*_]:
            raise NotImplementedError("multimodal output is not supported yet")

        case model:
            return _prepare_model_output_conversion(model)


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


def tools_as_tool_config(
    tools: Iterable[LMMToolSpecification] | None,
    /,
    *,
    tool_selection: LMMToolSelection,
) -> dict[str, Any] | None:
    toolChoice: dict[str, Any]
    match tool_selection:
        case "auto":
            toolChoice = {"auto": {}}

        case "required":
            toolChoice = {"any": {}}

        case "none":
            return None

        case tool:
            toolChoice = {"tool": {"name": tool}}

    tools_list = [{"toolSpec": convert_tool(tool)} for tool in tools or ()]
    if not tools_list:
        return None

    return {
        "tools": tools_list,
        "toolChoice": toolChoice,
    }
