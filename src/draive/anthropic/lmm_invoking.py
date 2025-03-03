from collections.abc import Iterable, Sequence
from typing import Any, cast

from anthropic import NOT_GIVEN
from anthropic.types import (
    Message,
    TextBlock,
    ToolUseBlock,
)
from haiway import MISSING, ArgumentsTrace, ResultTrace, as_list, ctx

from draive.anthropic.api import AnthropicAPI
from draive.anthropic.config import AnthropicConfig
from draive.anthropic.lmm import (
    content_block_as_content_element,
    context_element_as_message,
    convert_content_element,
    output_as_response_declaration,
    thinking_budget_as_config,
    tools_as_tool_config,
)
from draive.anthropic.types import AnthropicException
from draive.anthropic.utils import unwrap_missing
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
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
)

__all__ = [
    "AnthropicLMMInvoking",
]


class AnthropicLMMInvoking(AnthropicAPI):
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
        prefill: Multimodal | None = None,
        config: AnthropicConfig | None = None,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("anthropic_lmm_invocation"):
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
            completion_config: AnthropicConfig = config or ctx.state(AnthropicConfig).updated(
                **extra
            )
            ctx.record(completion_config)

            messages: list = [context_element_as_message(element) for element in context]

            response_prefill, output_decoder = output_as_response_declaration(
                output=output,
                prefill=prefill,
            )

            if response_prefill:  # use custom prefill
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            convert_content_element(element) for element in response_prefill.parts
                        ],
                    }
                )

            tool_choice, tools_list = tools_as_tool_config(
                tools,
                tool_selection=tool_selection,
            )

            completion: Message = await self._client.messages.create(
                model=completion_config.model,
                system=Instruction.formatted(instruction) if instruction else NOT_GIVEN,
                messages=messages,
                temperature=completion_config.temperature,
                top_p=unwrap_missing(completion_config.top_p),
                max_tokens=completion_config.max_tokens,
                thinking=thinking_budget_as_config(completion_config.thinking_tokens_budget),
                tools=tools_list,
                tool_choice=tool_choice,
                stop_sequences=as_list(cast(Sequence[str], completion_config.stop_sequences))
                if completion_config.stop_sequences is not MISSING
                else NOT_GIVEN,
                stream=False,
            )

            ctx.record(
                TokenUsage.for_model(
                    completion.model,
                    input_tokens=completion.usage.input_tokens,
                    # TODO: should we count cache_creation_input_tokens as well?
                    cached_tokens=completion.usage.cache_read_input_tokens,
                    output_tokens=completion.usage.output_tokens,
                ),
            )

            match completion.stop_reason:
                case "end_turn" | "tool_use" | "stop_sequence":
                    pass  # process results

                case "max_tokens":
                    raise AnthropicException(
                        "Invalid Anthropic completion - exceeded maximum length!",
                        completion,
                    )

                case _:
                    raise AnthropicException(
                        "Anthropic completion generation failed!",
                        completion,
                    )

            message_parts: list[TextBlock] = []
            tool_calls: list[ToolUseBlock] = []
            for part in completion.content:
                match part:
                    case TextBlock() as text:
                        message_parts.append(text)

                    case ToolUseBlock() as call:
                        tool_calls.append(call)

                    case _:
                        # TODO: add thinking support
                        pass  # skip other elements

            lmm_completion: LMMCompletion | None
            if message_parts:
                if response_prefill:
                    lmm_completion = LMMCompletion.of(
                        output_decoder(
                            MultimodalContent.of(
                                response_prefill,
                                *[content_block_as_content_element(part) for part in message_parts],
                            )
                        ),
                    )

                else:
                    lmm_completion = LMMCompletion.of(
                        output_decoder(
                            MultimodalContent.of(
                                *[content_block_as_content_element(part) for part in message_parts],
                            )
                        ),
                    )

            else:
                lmm_completion = None

            if tool_calls:
                assert tools, "Requesting tool call without tools"  # nosec: B101
                completion_tool_calls = LMMToolRequests(
                    completion=lmm_completion,
                    requests=[
                        LMMToolRequest(
                            identifier=call.id,
                            tool=call.name,
                            arguments=cast(dict[str, Any], call.input),
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
                raise AnthropicException(
                    "Invalid Anthropic completion, missing content!", completion
                )
