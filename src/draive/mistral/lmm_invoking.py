import json
from collections.abc import Iterable
from itertools import chain
from typing import Any
from uuid import uuid4

from haiway import ArgumentsTrace, ResultTrace, as_list, ctx, not_missing
from mistralai.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    MessagesTypedDict,
)
from mistralai.types import UNSET

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
from draive.metrics import TokenUsage
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralChatConfig
from draive.mistral.lmm import (
    content_chunk_as_content_element,
    content_element_as_content_chunk,
    context_element_as_messages,
    output_as_response_format,
    tools_as_tool_config,
)
from draive.mistral.types import MistralException
from draive.multimodal import MultimodalContent

__all__ = [
    "MistralLMMInvoking",
]


class MistralLMMInvoking(MistralAPI):
    def lmm_invoking(self) -> LMMInvocation:
        return LMMInvocation(invoke=self.lmm_invocation)

    async def lmm_invocation(  # noqa: C901, PLR0912, PLR0913
        self,
        *,
        instruction: Instruction | str | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        prefill: MultimodalContent | None = None,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("mistral_lmm_invocation"):
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
            config: MistralChatConfig = ctx.state(MistralChatConfig).updated(**extra)
            ctx.record(config)

            messages: list[MessagesTypedDict] = list(
                chain.from_iterable([context_element_as_messages(element) for element in context])
            )

            if prefill:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            content_element_as_content_chunk(element) for element in prefill.parts
                        ],
                        "prefix": True,
                    }
                )

            if instruction:
                messages = [
                    {
                        "role": "system",
                        "content": Instruction.formatted(instruction),
                    },
                    *messages,
                ]

            tool_choice, tools_list = tools_as_tool_config(
                tools,
                tool_selection=tool_selection,
            )

            completion: ChatCompletionResponse = await self._client.chat.complete_async(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p if not_missing(config.top_p) else None,
                max_tokens=config.max_tokens if not_missing(config.max_tokens) else UNSET,
                stream=False,
                stop=as_list(config.stop_sequences) if not_missing(config.stop_sequences) else None,
                random_seed=config.seed if not_missing(config.seed) else UNSET,
                response_format=output_as_response_format(output),
                tools=tools_list,
                tool_choice=tool_choice,
            )
            if usage := completion.usage:
                ctx.record(
                    TokenUsage.for_model(
                        completion.model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                    ),
                )

            if not completion.choices:
                raise MistralException("Invalid Mistral completion - missing choices!", completion)

            completion_choice: ChatCompletionChoice = completion.choices[0]

            match completion_choice.finish_reason:
                case "stop" | "tool_calls":
                    pass  # process results

                case "length":
                    raise MistralException(
                        "Invalid Mistral completion - exceeded maximum length!",
                        completion,
                    )

                case "error":
                    raise MistralException(
                        "Mistral completion generation failed!",
                        completion,
                    )

            completion_message: AssistantMessage = completion_choice.message

            lmm_completion: LMMCompletion | None
            if content := completion_message.content:
                match content:
                    case str() as string:
                        lmm_completion = LMMCompletion.of(string)

                    case chunks:
                        lmm_completion = LMMCompletion.of(
                            *[content_chunk_as_content_element(chunk) for chunk in chunks]
                        )

            else:
                lmm_completion = None

            if tool_calls := completion_message.tool_calls:
                assert tools, "Requesting tool call without tools"  # nosec: B101
                completion_tool_calls = LMMToolRequests(
                    completion=lmm_completion,
                    requests=[
                        LMMToolRequest(
                            identifier=call.id or uuid4().hex,
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
                raise MistralException("Invalid Mistral completion, missing content!", completion)
