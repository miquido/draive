from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, overload

from draive.metrics import ArgumentsTrace, ResultTrace
from draive.metrics.tokens import TokenUsage
from draive.ollama.client import OllamaClient
from draive.ollama.config import OllamaChatConfig
from draive.ollama.errors import OllamaException
from draive.ollama.models import ChatCompletionResponse, ChatMessage
from draive.parameters import ToolSpecification
from draive.scope import ctx
from draive.types import (
    Instruction,
    LMMCompletion,
    LMMCompletionChunk,
    LMMContextElement,
    LMMInput,
    LMMOutput,
    LMMOutputStream,
    LMMOutputStreamChunk,
    LMMToolRequests,
    LMMToolResponse,
)

__all__ = [
    "ollama_lmm_invocation",
]


@overload
async def ollama_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def ollama_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMOutput: ...


@overload
async def ollama_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def ollama_lmm_invocation(  # noqa: PLR0913
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput:
    with ctx.nested(
        "ollama_lmm_invocation",
        metrics=[
            ArgumentsTrace.of(
                instruction=instruction,
                context=context,
                tools=tools,
                tool_requirement=tool_requirement,
                output=output,
                stream=stream,
                **extra,
            ),
        ],
    ):
        ctx.log_debug("Requested Ollama lmm")
        client: OllamaClient = ctx.dependency(OllamaClient)
        config: OllamaChatConfig = ctx.state(OllamaChatConfig).updated(**extra)
        ctx.record(config)

        if tools:
            ctx.log_warning(
                "Attempting to use Ollama with tools which is not supported."
                " Ignoring provided tools..."
            )

        match output:
            case "text":
                config = config.updated(response_format="text")

            case "json":
                config = config.updated(response_format="json")

        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=Instruction.of(instruction).format(),
            ),
            *[_convert_context_element(element=element) for element in context],
        ]

        if stream:
            return ctx.stream(
                generator=_chat_completion_stream(
                    client=client,
                    config=config,
                    messages=messages,
                ),
            )

        else:
            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
            )


def _convert_context_element(
    element: LMMContextElement,
) -> ChatMessage:
    match element:
        case LMMInput() as input:
            return ChatMessage(
                role="user",
                content=input.content.as_string(),
            )

        case LMMCompletion() as completion:
            return ChatMessage(
                role="assistant",
                content=completion.content.as_string(),
            )

        case LMMToolRequests():
            raise NotImplementedError("Tools use is not supported by Ollama")

        case LMMToolResponse():
            raise NotImplementedError("Tools use is not supported by Ollama")


async def _chat_completion(
    *,
    client: OllamaClient,
    config: OllamaChatConfig,
    messages: list[ChatMessage],
) -> LMMOutput:
    completion: ChatCompletionResponse = await client.chat_completion(
        config=config,
        messages=messages,
    )

    ctx.record(
        TokenUsage.for_model(
            config.model,
            input_tokens=completion.prompt_eval_count,
            output_tokens=completion.eval_count,
        ),
    )

    if message := completion.message.content:
        ctx.record(ResultTrace.of(message))
        return LMMCompletion.of(message)

    else:
        raise OllamaException("Invalid Ollama completion", completion)


async def _chat_completion_stream(
    *,
    client: OllamaClient,
    config: OllamaChatConfig,
    messages: list[ChatMessage],
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    ctx.log_debug("Ollama streaming api is not supported yet, using regular response...")
    output: LMMOutput = await _chat_completion(
        client=client,
        config=config,
        messages=messages,
    )

    match output:
        case LMMCompletion() as completion:
            yield LMMCompletionChunk.of(completion.content)

        case other:
            yield other
