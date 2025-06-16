import json
from collections.abc import AsyncIterator
from itertools import chain
from typing import Any, Literal, overload
from uuid import uuid4

from haiway import ObservabilityLevel, ctx
from ollama import ChatResponse, Message, Options

from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMInstruction,
    LMMOutput,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMTools,
)
from draive.lmm.types import LMMStreamOutput
from draive.multimodal import MultimodalContent
from draive.ollama.api import OllamaAPI
from draive.ollama.config import OllamaChatConfig
from draive.ollama.lmm import (
    context_element_as_messages,
    output_as_response_declaration,
    tools_as_tool_config,
)
from draive.ollama.types import OllamaException
from draive.ollama.utils import unwrap_missing

__all__ = ("OllamaLMMGeneration",)


class OllamaLMMGeneration(OllamaAPI):
    def lmm(self) -> LMM:
        return LMM(completing=self.lmm_completion)

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        config: OllamaChatConfig | None = None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        config: OllamaChatConfig | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        config: OllamaChatConfig | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        if stream:
            raise NotImplementedError("ollama streaming is not implemented yet")

        chat_config: OllamaChatConfig = config or ctx.state(OllamaChatConfig)
        tools = tools or LMMTools.none
        with ctx.scope("ollama_lmm_completion", chat_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "lmm.provider": "ollama",
                    "lmm.model": chat_config.model,
                    "lmm.temperature": chat_config.temperature,
                    "lmm.max_tokens": chat_config.max_tokens,
                    "lmm.seed": chat_config.seed,
                    "lmm.tools": [tool["name"] for tool in tools.specifications],
                    "lmm.tool_selection": f"{tools.selection}",
                    "lmm.stream": stream,
                    "lmm.output": f"{output}",
                    "lmm.instruction": f"{instruction}",
                    "lmm.context": [element.to_str() for element in context],
                },
            )

            messages: list[Message] = list(
                chain.from_iterable([context_element_as_messages(element) for element in context])
            )

            if instruction:
                messages = [
                    Message(
                        role="system",
                        content=instruction,
                    ),
                    *messages,
                ]

            response_format, output_decoder = output_as_response_declaration(output)

            tools_list = tools_as_tool_config(
                tools.specifications,
                tool_selection=tools.selection,
            )

            completion: ChatResponse = await self._client.chat(
                model=chat_config.model,
                messages=messages,
                format=response_format,
                tools=tools_list,
                options=Options(
                    temperature=chat_config.temperature,
                    num_predict=unwrap_missing(chat_config.max_tokens),
                    top_k=unwrap_missing(chat_config.top_k),
                    top_p=unwrap_missing(chat_config.top_p),
                    seed=unwrap_missing(chat_config.seed),
                    stop=unwrap_missing(chat_config.stop_sequences),
                ),
                stream=False,
            )

            completion_message: Message = completion.message

            lmm_completion: LMMCompletion | None
            if content := completion_message.content:
                lmm_completion = LMMCompletion.of(output_decoder(MultimodalContent.of(content)))

            else:
                lmm_completion = None

            if tool_calls := completion_message.tool_calls:
                assert tools, "Requesting tool call without tools"  # nosec: B101
                completion_tool_calls = LMMToolRequests(
                    content=lmm_completion.content if lmm_completion else None,
                    requests=[
                        LMMToolRequest(
                            identifier=uuid4().hex,
                            tool=call.function.name,
                            arguments=json.loads(call.function.arguments)
                            if isinstance(call.function.arguments, str)
                            else call.function.arguments,
                        )
                        for call in tool_calls
                    ],
                )
                ctx.record(
                    ObservabilityLevel.INFO,
                    event="lmm.tool_requests",
                    attributes={"lmm.tools": [call.function.name for call in tool_calls]},
                )
                return completion_tool_calls

            elif lmm_completion:
                ctx.record(
                    ObservabilityLevel.INFO,
                    event="lmm.completion",
                )
                return lmm_completion

            else:
                raise OllamaException("Invalid Ollama completion, missing content!", completion)
