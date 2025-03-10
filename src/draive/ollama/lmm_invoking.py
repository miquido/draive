import json
from collections.abc import Iterable
from itertools import chain
from typing import Any
from uuid import uuid4

from haiway import ArgumentsTrace, ResultTrace, ctx
from ollama import ChatResponse, Message, Options

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContext,
    LMMInvocation,
    LMMOutput,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolSelection,
    LMMToolSpecification,
)
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

__all__ = [
    "OllamaLMMInvoking",
]


class OllamaLMMInvoking(OllamaAPI):
    def lmm_invoking(self) -> LMMInvocation:
        return LMMInvocation(invoke=self.lmm_invocation)

    async def lmm_invocation(  # noqa: PLR0913
        self,
        *,
        instruction: Instruction | str | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        config: OllamaChatConfig | None = None,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("ollama_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                )
            )
            chat_config: OllamaChatConfig = config or ctx.state(OllamaChatConfig).updated(**extra)
            ctx.record(chat_config)

            messages: list[Message] = list(
                chain.from_iterable([context_element_as_messages(element) for element in context])
            )

            if instruction:
                messages = [
                    Message(
                        role="system",
                        content=Instruction.formatted(instruction),
                    ),
                    *messages,
                ]

            response_format, output_decoder = output_as_response_declaration(output)

            tools_list = tools_as_tool_config(
                tools,
                tool_selection=tool_selection,
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
                ctx.record(ResultTrace.of(completion_tool_calls))
                return completion_tool_calls

            elif lmm_completion:
                ctx.record(ResultTrace.of(lmm_completion))
                return lmm_completion

            else:
                raise OllamaException("Invalid Ollama completion, missing content!", completion)
