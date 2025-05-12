from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from haiway import as_dict
from ollama import Image, Message, Tool
from pydantic.json_schema import JsonSchemaValue

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
    MultimodalContent,
)
from draive.multimodal.media import MediaData
from draive.ollama.types import OllamaException
from draive.parameters import DataModel

__all__ = (
    "context_element_as_messages",
    "output_as_response_declaration",
    "tools_as_tool_config",
)


def context_element_as_messages(
    element: LMMContextElement,
    /,
) -> Iterable[Message]:
    match element:
        case LMMInput() as input:
            return (
                Message(
                    role="user",
                    content=element.content.without_media().to_str(),
                    images=[
                        Image(value=image.data)
                        if isinstance(image, MediaData)
                        else Image(value=image.uri)
                        for image in input.content.media("image")
                    ],
                ),
            )

        case LMMCompletion() as completion:
            return (
                Message(
                    role="assistant",
                    content=element.content.without_media().to_str(),
                    images=[
                        Image(value=image.data)
                        if isinstance(image, MediaData)
                        else Image(value=image.uri)
                        for image in completion.content.media("image")
                    ],
                ),
            )

        case LMMToolRequests() as tool_requests:
            return (
                Message(
                    role="assistant",
                    # images=[Image(value=image.) for image in element.content.media("image")],
                    tool_calls=[
                        Message.ToolCall(
                            function=Message.ToolCall.Function(
                                name=request.tool,
                                arguments=as_dict(request.arguments),
                            )
                        )
                        for request in tool_requests.requests
                    ],
                ),
            )

        case LMMToolResponses() as tool_responses:
            return (
                Message(
                    role="assistant",
                    content=response.content.without_media().to_str(),
                    images=[
                        Image(value=image.data)
                        if isinstance(image, MediaData)
                        else Image(value=image.uri)
                        for image in response.content.media("image")
                    ],
                )
                for response in tool_responses.responses
            )


def tool_specification_as_tool(
    tool: LMMToolSpecification,
    /,
) -> Tool:
    return Tool(
        type="function",
        function=Tool.Function(
            name=tool["name"],
            description=tool["description"],
            parameters=cast(Tool.Function.Parameters, tool["parameters"]),
        ),
    )


def output_as_response_declaration(
    output: LMMOutputSelection,
) -> tuple[
    Literal["json"] | JsonSchemaValue | None, Callable[[MultimodalContent], MultimodalContent]
]:
    match output:
        case "auto":
            return (None, _auto_output_conversion)

        case ["text"] | "text":
            return (None, _text_output_conversion)

        case "json":
            return ("json", _json_output_conversion)

        case "image":
            raise NotImplementedError("image output is not supported by Ollama")

        case "audio":
            raise NotImplementedError("audio output is not supported by Ollama")

        case "video":
            raise NotImplementedError("video output is not supported by Ollama")

        case [*_]:
            raise NotImplementedError("multimodal output is not supported by Ollama")

        case model:
            return (
                cast(dict[str, Any], model.__PARAMETERS_SPECIFICATION__),
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
) -> list[Tool] | None:
    tools_list: list[Tool] = [tool_specification_as_tool(tool) for tool in (tools or [])]
    if not tools_list:
        return None

    match tool_selection:
        case "auto":
            return tools_list

        case "none":
            return None

        case "required":
            raise OllamaException("Tool requirement is not supported in Ollama")

        case _:
            raise OllamaException("Tool suggestions are not supported in Ollama")
