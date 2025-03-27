import json
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import MISSING, ArgumentsTrace, ResultTrace, as_list, ctx
from openai import NOT_GIVEN
from openai import RateLimitError as OpenAIRateLimitError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

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
from draive.multimodal.content import MultimodalContent
from draive.utils import RateLimitError
from draive.vllm.api import VLLMAPI
from draive.vllm.config import VLLMChatConfig
from draive.vllm.lmm import (
    context_element_as_messages,
    output_as_response_declaration,
    tools_as_tool_config,
)
from draive.vllm.types import VLLMException
from draive.vllm.utils import unwrap_missing

__all__ = [
    "VLLMLMMInvoking",
]


class VLLMLMMInvoking(VLLMAPI):
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
        config: VLLMChatConfig | None = None,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("vllm_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    tool_selection=tool_selection,
                    tools=tools,
                    output=output,
                    **extra,
                ),
            )
            chat_config: VLLMChatConfig = config or ctx.state(VLLMChatConfig).updated(**extra)
            ctx.record(chat_config)

            messages: list[ChatCompletionMessageParam] = list(
                chain.from_iterable(
                    [
                        context_element_as_messages(
                            element,
                            config=chat_config,
                        )
                        for element in context
                    ]
                )
            )

            if instruction:
                messages = [
                    {
                        "role": "system",
                        "content": Instruction.formatted(instruction),
                    },
                    *messages,
                ]

            response_format, response_modalities, output_decoder = output_as_response_declaration(
                output
            )

            tool_choice, tools_list = tools_as_tool_config(tools, tool_selection=tool_selection)

            completion: ChatCompletion
            try:
                completion = await self._client.chat.completions.create(
                    messages=messages,
                    model=chat_config.model,
                    frequency_penalty=unwrap_missing(chat_config.frequency_penalty),
                    max_tokens=unwrap_missing(chat_config.max_tokens),
                    n=1,
                    response_format=response_format,
                    seed=unwrap_missing(chat_config.seed),
                    stream=False,
                    temperature=chat_config.temperature,
                    tools=tools_list,
                    tool_choice=tool_choice,
                    parallel_tool_calls=unwrap_missing(chat_config.parallel_tool_calls)
                    if tools_list
                    else NOT_GIVEN,
                    top_p=unwrap_missing(chat_config.top_p),
                    timeout=unwrap_missing(chat_config.timeout),
                    stop=as_list(cast(Sequence[str], chat_config.stop_sequences))
                    if chat_config.stop_sequences is not MISSING
                    else NOT_GIVEN,
                )

            except OpenAIRateLimitError as exc:  # retry on rate limit after delay
                if delay := exc.response.headers.get("Retry-After"):
                    try:
                        raise RateLimitError(retry_after=float(delay)) from exc

                    except ValueError:
                        raise exc from None

                else:
                    raise exc

            if usage := completion.usage:
                ctx.record(
                    TokenUsage.for_model(
                        completion.model,
                        input_tokens=usage.prompt_tokens,
                        cached_tokens=None,
                        output_tokens=usage.completion_tokens,
                    ),
                )

            if not completion.choices:
                raise VLLMException("Invalid VLLM completion - missing messages!", completion)

            completion_choice = completion.choices[0]
            match completion_choice.finish_reason:
                case "stop" | "tool_calls":
                    pass  # process results

                case "length":
                    raise VLLMException(
                        "Invalid VLLM completion - exceeded maximum length!",
                        completion,
                    )

                case "error":
                    raise VLLMException(
                        "VLLM completion generation failed!",
                        completion,
                    )

            completion_message: ChatCompletionMessage = completion_choice.message

            lmm_completion: LMMCompletion | None
            if content := completion_message.content:
                # TODO: add audio support
                lmm_completion = LMMCompletion.of(output_decoder(MultimodalContent.of(content)))

            else:
                lmm_completion = None

            if tool_calls := completion_message.tool_calls:
                assert tools, "Requesting tool call without tools"  # nosec: B101
                completion_tool_calls = LMMToolRequests(
                    content=lmm_completion.content if lmm_completion else None,
                    requests=[
                        LMMToolRequest(
                            identifier=call.id,
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
                raise VLLMException("Invalid VLLM completion, missing content!", completion)
