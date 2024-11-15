from collections.abc import Iterable
from typing import Any, Literal

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInvocation,
    LMMOutput,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolSelection,
    ToolSpecification,
)
from draive.metrics.tokens import TokenUsage
from draive.ollama.client import OllamaClient
from draive.ollama.config import OllamaChatConfig
from draive.ollama.models import ChatCompletionResponse, ChatMessage
from draive.parameters import ParametersSpecification

__all__ = [
    "ollama_lmm",
]


def ollama_lmm(
    client: OllamaClient | None = None,
    /,
) -> LMMInvocation:
    client = client or OllamaClient.shared()

    async def lmm_invocation(
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("ollama_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    tool_selection=tool_selection,
                    tools=tools,
                    output=output,
                    **extra,
                )
            )

            config: OllamaChatConfig = ctx.state(OllamaChatConfig).updated(**extra)
            ctx.record(config)

            if tools:
                ctx.log_warning(
                    "Attempting to use Ollama with tools which is not supported."
                    " Ignoring provided tools..."
                )

            match output:
                case "auto" | "text":
                    config = config.updated(response_format="text")

                case _:
                    config = config.updated(response_format="json")

            messages: list[ChatMessage] = [
                _convert_context_element(element=element) for element in context
            ]

            if instruction:
                messages = [
                    ChatMessage(
                        role="system",
                        content=Instruction.of(instruction).format(),
                    ),
                    *messages,
                ]

            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
            )

    return LMMInvocation(invoke=lmm_invocation)


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

    ctx.record(ResultTrace.of(completion.message.content))
    return LMMCompletion.of(completion.message.content)
