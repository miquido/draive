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
    Multimodal,
    MultimodalContent,
)
from draive.ollama.types import OllamaException
from draive.parameters import DataModel

__all__ = [
    "context_element_as_messages",
    "output_as_response_declaration",
    "tools_as_tool_config",
]


def context_element_as_messages(
    element: LMMContextElement,
    /,
) -> Iterable[Message]:
    match element:
        case LMMInput() as input:
            return (
                Message(
                    role="user",
                    content=element.content.without_media().as_string(),
                    images=[Image(value=image.source) for image in input.content.media("image")],
                ),
            )

        case LMMCompletion() as completion:
            return (
                Message(
                    role="assistant",
                    content=element.content.without_media().as_string(),
                    images=[
                        Image(value=image.source) for image in completion.content.media("image")
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
                    content=response.content.without_media().as_string(),
                    images=[Image(value=image.source) for image in response.content.media("image")],
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
) -> tuple[Literal["json"] | JsonSchemaValue | None, Callable[[MultimodalContent], Multimodal]]:
    match output:
        case "auto":
            return (None, _auto_output_conversion)

        case "text":
            return (None, _text_output_conversion)

        case "json":
            return ("json", _json_output_conversion)

        case "image":
            raise NotImplementedError("image output is not supported by Ollama")

        case "audio":
            raise NotImplementedError("audio output is not supported by Ollama")

        case "video":
            raise NotImplementedError("video output is not supported by Ollama")

        case model:
            return (
                cast(dict[str, Any], model.__PARAMETERS_SPECIFICATION__),
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


# from collections.abc import Iterable
# from typing import Any

# from haiway import ArgumentsTrace, ResultTrace, ctx

# from draive.instructions import Instruction
# from draive.lmm import (
#     LMMCompletion,
#     LMMContext,
#     LMMContextElement,
#     LMMInput,
#     LMMInvocation,
#     LMMOutput,
#     LMMOutputSelection,
#     LMMToolRequests,
#     LMMToolResponses,
#     LMMToolSelection,
#     LMMToolSpecification,
# )
# from draive.metrics.tokens import TokenUsage
# from draive.ollama.client import OllamaClient
# from draive.ollama.config import OllamaChatConfig
# from draive.ollama.models import ChatCompletionResponse, ChatMessage

# __all__ = [
#     "ollama_lmm",
# ]


# def ollama_lmm(
#     client: OllamaClient | None = None,
#     /,
# ) -> LMMInvocation:
#     client = client or OllamaClient.shared()

#     async def lmm_invocation(
#         *,
#         instruction: Instruction | str | None,
#         context: LMMContext,
#         tool_selection: LMMToolSelection,
#         tools: Iterable[LMMToolSpecification] | None,
#         output: LMMOutputSelection,
#         **extra: Any,
#     ) -> LMMOutput:
#         with ctx.scope("ollama_lmm_invocation"):
#             ctx.record(
#                 ArgumentsTrace.of(
#                     instruction=instruction,
#                     context=context,
#                     tool_selection=tool_selection,
#                     tools=tools,
#                     output=output,
#                     **extra,
#                 )
#             )

#             config: OllamaChatConfig = ctx.state(OllamaChatConfig).updated(**extra)
#             ctx.record(config)

#             if tools:
#                 ctx.log_warning(
#                     "Attempting to use Ollama with tools which is not supported."
#                     " Ignoring provided tools..."
#                 )

#             match output:
#                 case "auto" | "text":
#                     config = config.updated(response_format="text")

#                 case "image":
#                     raise NotImplementedError("image output is not supported by ollama")

#                 case "audio":
#                     raise NotImplementedError("audio output is not supported by ollama")

#                 case "video":
#                     raise NotImplementedError("video output is not supported by ollama")

#                 case _:
#                     config = config.updated(response_format="json")

#             messages: list[ChatMessage] = [
#                 _convert_context_element(element=element) for element in context
#             ]

#             if instruction:
#                 messages = [
#                     ChatMessage(
#                         role="system",
#                         content=Instruction.of(instruction).format(),
#                     ),
#                     *messages,
#                 ]

#             return await _chat_completion(
#                 client=client,
#                 config=config,
#                 messages=messages,
#             )

#     return LMMInvocation(invoke=lmm_invocation)


# def _convert_context_element(
#     element: LMMContextElement,
# ) -> ChatMessage:
#     match element:
#         case LMMInput() as input:
#             return ChatMessage(
#                 role="user",
#                 content=input.content.as_string(),
#             )

#         case LMMCompletion() as completion:
#             return ChatMessage(
#                 role="assistant",
#                 content=completion.content.as_string(),
#             )

#         case LMMToolRequests():
#             raise NotImplementedError("Tools use is not supported by Ollama")

#         case LMMToolResponses():
#             raise NotImplementedError("Tools use is not supported by Ollama")


# async def _chat_completion(
#     *,
#     client: OllamaClient,
#     config: OllamaChatConfig,
#     messages: list[ChatMessage],
# ) -> LMMOutput:
#     completion: ChatCompletionResponse = await client.chat_completion(
#         config=config,
#         messages=messages,
#     )

#     ctx.record(
#         TokenUsage.for_model(
#             config.model,
#             input_tokens=completion.prompt_eval_count,
#             cached_tokens=None,
#             output_tokens=completion.eval_count,
#         ),
#     )

#     ctx.record(ResultTrace.of(completion.message.content))
#     return LMMCompletion.of(completion.message.content)
