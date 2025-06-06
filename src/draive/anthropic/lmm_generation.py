from collections.abc import AsyncIterator, Sequence
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
from haiway import MISSING, ObservabilityLevel, as_list, ctx

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
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMInstruction,
    LMMOutput,
    LMMOutputSelection,
    LMMStreamOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMTools,
)
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
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        config: AnthropicConfig | None = None,
        prefill: Multimodal | None = None,
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
        stream: Literal[True],
        config: AnthropicConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: bool = False,
        config: AnthropicConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        if stream:
            raise NotImplementedError("Anthropic streaming is not implemented yet")

        completion_config: AnthropicConfig = config or ctx.state(AnthropicConfig)
        tools = tools or LMMTools.none
        with ctx.scope("anthropic_lmm_completion", completion_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "lmm.provider": "anthropic",
                    "lmm.model": completion_config.model,
                    "lmm.temperature": completion_config.temperature,
                    "lmm.max_tokens": completion_config.max_tokens,
                    "lmm.tools": [tool["name"] for tool in tools.specifications],
                    "lmm.tool_selection": f"{tools.selection}",
                    "lmm.stream": stream,
                    "lmm.output": f"{output}",
                    "lmm.instruction": f"{instruction}",
                    "lmm.context": [element.to_str() for element in context],
                },
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
                tools.specifications,
                tool_selection=tools.selection,
            )

            completion: Message
            try:
                completion = await self._client.messages.create(
                    model=completion_config.model,
                    system=instruction if instruction else NOT_GIVEN,
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
                    ctx.record(
                        ObservabilityLevel.WARNING,
                        event="lmm.rate_limit",
                        attributes={"delay": delay},
                    )
                    try:
                        raise RateLimitError(retry_after=float(delay)) from exc

                    except ValueError:
                        raise exc from None

                else:
                    ctx.record(
                        ObservabilityLevel.WARNING,
                        event="lmm.rate_limit",
                    )
                    raise exc

            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.input_tokens",
                value=completion.usage.input_tokens,
                unit="tokens",
                attributes={"lmm.model": completion.model},
            )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.input_tokens.cached",
                value=completion.usage.cache_creation_input_tokens or 0,
                unit="tokens",
                attributes={"lmm.model": completion.model},
            )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.output_tokens",
                value=completion.usage.output_tokens,
                unit="tokens",
                attributes={"lmm.model": completion.model},
            )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.output_tokens.cached",
                value=completion.usage.cache_read_input_tokens or 0,
                unit="tokens",
                attributes={"lmm.model": completion.model},
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
                ctx.record(
                    ObservabilityLevel.INFO,
                    event="lmm.tool_requests",
                    attributes={"lmm.tools": [call.name for call in tool_calls]},
                )
                return completion_tool_calls

            elif lmm_completion:
                ctx.record(
                    ObservabilityLevel.INFO,
                    event="lmm.completion",
                )
                return lmm_completion

            else:
                raise AnthropicException(
                    "Invalid Anthropic completion, missing content!", completion
                )
