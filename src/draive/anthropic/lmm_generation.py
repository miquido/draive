from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Literal, cast, overload

from anthropic import NOT_GIVEN
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic.types import (
    Message,
    MessageParam,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
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
from draive.commons import META_EMPTY
from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContext,
    LMMOutput,
    LMMOutputSelection,
    LMMStreamOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.lmm.state import LMM
from draive.metrics import TokenUsage
from draive.multimodal import Multimodal, MultimodalContent
from draive.utils import RateLimitError

__all__ = ("AnthropicLMMGeneration",)


class AnthropicLMMGeneration(AnthropicAPI):
    def lmm(self) -> LMM:
        return LMM(completing=self.lmm_completion)

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        prefill: Multimodal | None = None,
        config: AnthropicConfig | None = None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        prefill: Multimodal | None = None,
        config: AnthropicConfig | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        prefill: Multimodal | None = None,
        config: AnthropicConfig | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        if stream:
            raise NotImplementedError("Anthropic streaming is not implemented yet")

        with ctx.scope("anthropic_lmm_completion"):
            completion_config: AnthropicConfig = config or ctx.state(AnthropicConfig).updated(
                **extra
            )
            ctx.record(
                ArgumentsTrace.of(
                    config=completion_config,
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                )
            )

            messages: list[MessageParam] = [
                context_element_as_message(element) for element in context
            ]

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

            completion: Message
            try:
                completion = await self._client.messages.create(
                    model=completion_config.model,
                    system=Instruction.formatted(instruction) if instruction else NOT_GIVEN,
                    messages=messages,
                    temperature=completion_config.temperature,
                    top_p=unwrap_missing(completion_config.top_p),
                    max_tokens=completion_config.max_tokens,
                    thinking=thinking_budget_as_config(completion_config.thinking_tokens_budget)
                    if self._provider == "anthropic"
                    else NOT_GIVEN,  # bedrock does not support thinking yet
                    tools=tools_list,
                    tool_choice=tool_choice,
                    stop_sequences=as_list(cast(Sequence[str], completion_config.stop_sequences))
                    if completion_config.stop_sequences is not MISSING
                    else NOT_GIVEN,
                    stream=False,
                )

            except AnthropicRateLimitError as exc:  # retry on rate limit after delay
                if delay := exc.response.headers.get("Retry-After"):
                    try:
                        raise RateLimitError(retry_after=float(delay)) from exc

                    except ValueError:
                        raise exc from None

                else:
                    raise exc

            ctx.record(
                TokenUsage.for_model(
                    completion.model,
                    # NOTE: cache input tokens are charged extra
                    # we are not handling it that specific though
                    input_tokens=completion.usage.input_tokens
                    + (completion.usage.cache_creation_input_tokens or 0),
                    cached_tokens=completion.usage.cache_read_input_tokens,
                    output_tokens=completion.usage.output_tokens,
                ),
            )

            match completion.stop_reason:
                case "end_turn" | "tool_use" | "stop_sequence":
                    pass  # process results

                case "max_tokens":
                    raise AnthropicException("Invalid Anthropic completion - tokens limit!")

                case reason:
                    raise AnthropicException(f"Anthropic completion generation failed - {reason}!")

            message_parts: list[TextBlock | ThinkingBlock | RedactedThinkingBlock] = []
            tool_calls: list[ToolUseBlock] = []
            for part in completion.content:
                match part:
                    case TextBlock() as text:
                        message_parts.append(text)

                    case ToolUseBlock() as call:
                        tool_calls.append(call)

                    case ThinkingBlock() as thinking:
                        message_parts.append(thinking)

                    case RedactedThinkingBlock() as redacted_thinking:
                        message_parts.append(redacted_thinking)

                    case _:
                        pass  # skip unknown elements

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
                    content=lmm_completion.content if lmm_completion else None,
                    requests=[
                        LMMToolRequest(
                            identifier=call.id,
                            tool=call.name,
                            arguments=cast(dict[str, Any], call.input),
                        )
                        for call in tool_calls
                    ],
                    meta=META_EMPTY,
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
