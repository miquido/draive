import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Sequence
from itertools import chain
from typing import Any, Literal, cast, overload
from uuid import uuid4

from haiway import MISSING, ObservabilityLevel, as_list, ctx
from openai import NOT_GIVEN, AsyncStream, NotGiven
from openai import RateLimitError as OpenAIRateLimitError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from openai.types.chat.completion_create_params import ResponseFormat

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
from draive.lmm.types import LMMStreamChunk, LMMStreamOutput
from draive.multimodal import MultimodalContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIChatConfig
from draive.openai.lmm import (
    context_element_as_messages,
    output_as_response_declaration,
    tools_as_tool_config,
)
from draive.openai.types import OpenAIException
from draive.openai.utils import unwrap_missing
from draive.utils import RateLimitError

__all__ = ("OpenAILMMGeneration",)


class OpenAILMMGeneration(OpenAIAPI):
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
        config: OpenAIChatConfig | None = None,
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
        config: OpenAIChatConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: bool = False,
        config: OpenAIChatConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        tools = tools or LMMTools.none
        completion_config: OpenAIChatConfig = config or ctx.state(OpenAIChatConfig).updated(**extra)
        with ctx.scope("openai_lmm_completion", completion_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "lmm.provider": "openai",
                    "lmm.model": completion_config.model,
                    "lmm.temperature": completion_config.temperature,
                    "lmm.max_tokens": completion_config.max_tokens,
                    "lmm.seed": completion_config.seed,
                    "lmm.vision_details": completion_config.vision_details,
                    "lmm.tools": [tool["name"] for tool in tools.specifications],
                    "lmm.tool_selection": f"{tools.selection}",
                    "lmm.stream": stream,
                    "lmm.output": f"{output}",
                    "lmm.instruction": f"{instruction}",
                    "lmm.context": [element.to_str() for element in context],
                },
            )

            messages: list[ChatCompletionMessageParam] = list(
                chain.from_iterable(
                    [
                        context_element_as_messages(
                            element,
                            vision_details=completion_config.vision_details,
                        )
                        for element in context
                    ]
                )
            )

            if instruction:
                messages = [
                    {
                        "role": "system",
                        "content": instruction,
                    },
                    *messages,
                ]

            response_format, output_decoder = output_as_response_declaration(output)

            tool_choice, tools_list = tools_as_tool_config(
                tools.specifications,
                tool_selection=tools.selection,
            )

            if stream:
                return await self._completion_stream(
                    model=completion_config.model,
                    temperature=completion_config.temperature,
                    top_p=unwrap_missing(completion_config.top_p),
                    frequency_penalty=unwrap_missing(completion_config.frequency_penalty),
                    messages=messages,
                    tools=tools_list,
                    tool_choice=tool_choice,
                    parallel_tool_calls=unwrap_missing(completion_config.parallel_tool_calls)
                    if tools_list
                    else NOT_GIVEN,
                    max_tokens=unwrap_missing(completion_config.max_tokens),
                    response_format=response_format,
                    seed=unwrap_missing(completion_config.seed),
                    stop=as_list(cast(Sequence[str], completion_config.stop_sequences))
                    if completion_config.stop_sequences is not MISSING
                    else NOT_GIVEN,
                    timeout=unwrap_missing(completion_config.timeout),
                    output_decoder=output_decoder,
                )

            else:
                return await self._completion(
                    model=completion_config.model,
                    temperature=completion_config.temperature,
                    top_p=unwrap_missing(completion_config.top_p),
                    frequency_penalty=unwrap_missing(completion_config.frequency_penalty),
                    messages=messages,
                    tools=tools_list,
                    tool_choice=tool_choice,
                    parallel_tool_calls=unwrap_missing(completion_config.parallel_tool_calls)
                    if tools_list
                    else NOT_GIVEN,
                    max_tokens=unwrap_missing(completion_config.max_tokens),
                    response_format=response_format,
                    seed=unwrap_missing(completion_config.seed),
                    stop=as_list(cast(Sequence[str], completion_config.stop_sequences))
                    if completion_config.stop_sequences is not MISSING
                    else NOT_GIVEN,
                    timeout=unwrap_missing(completion_config.timeout),
                    output_decoder=output_decoder,
                )

    async def _completion(  # noqa: C901, PLR0912, PLR0913
        self,
        model: str,
        temperature: float,
        top_p: float | NotGiven,
        frequency_penalty: float | NotGiven,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Iterable[ChatCompletionToolParam] | NotGiven,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven,
        parallel_tool_calls: bool | NotGiven,
        max_tokens: int | NotGiven,
        response_format: ResponseFormat | NotGiven,
        seed: int | NotGiven,
        stop: list[str] | NotGiven,
        timeout: float | NotGiven,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> LMMOutput:
        completion: ChatCompletion
        try:
            completion = await self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                n=1,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=seed,
                stop=stop,
                timeout=timeout,
                stream=False,
            )

        except OpenAIRateLimitError as exc:  # retry on rate limit after delay
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

        if usage := completion.usage:
            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.input_tokens",
                value=usage.prompt_tokens,
                unit="tokens",
                attributes={"lmm.model": completion.model},
            )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="lmm.output_tokens",
                value=usage.completion_tokens,
                unit="tokens",
                attributes={"lmm.model": completion.model},
            )

        if fingerprint := completion.system_fingerprint:
            ctx.log_debug(f"OpenAI system fingerprint:{fingerprint}")

        if not completion.choices:
            raise OpenAIException("Invalid OpenAI completion - missing messages!", completion)

        completion_choice = completion.choices[0]
        match completion_choice.finish_reason:
            case "stop" | "tool_calls":
                pass  # process results

            case "length":
                raise OpenAIException(
                    "Invalid OpenAI completion - exceeded maximum length!",
                    completion,
                )

            case "error":
                raise OpenAIException(
                    "OpenAI completion generation failed!",
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
            raise OpenAIException("Invalid OpenAI completion, missing content!", completion)

    async def _completion_stream(  # noqa: C901, PLR0913, PLR0915
        self,
        model: str,
        temperature: float,
        top_p: float | NotGiven,
        frequency_penalty: float | NotGiven,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Iterable[ChatCompletionToolParam] | NotGiven,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven,
        parallel_tool_calls: bool | NotGiven,
        max_tokens: int | NotGiven,
        response_format: ResponseFormat | NotGiven,
        seed: int | NotGiven,
        stop: list[str] | NotGiven,
        timeout: float | NotGiven,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> AsyncIterator[LMMStreamOutput]:
        accumulated_tool_calls: list[ChoiceDeltaToolCall] = []
        completion_stream: AsyncStream[ChatCompletionChunk]
        try:
            completion_stream: AsyncStream[
                ChatCompletionChunk
            ] = await self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                n=1,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=seed,
                stop=stop,
                timeout=timeout,
                stream_options={"include_usage": True},
                stream=True,
            )

        except OpenAIRateLimitError as exc:  # retry on rate limit after delay
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

        async def stream() -> AsyncGenerator[LMMStreamOutput]:  # noqa: C901, PLR0912, PLR0915
            async for part in completion_stream:
                if usage := part.usage:  # record usage if able (expected in the last part)
                    ctx.record(
                        ObservabilityLevel.INFO,
                        metric="lmm.input_tokens",
                        value=usage.prompt_tokens,
                        unit="tokens",
                        attributes={"lmm.model": part.model},
                    )
                    ctx.record(
                        ObservabilityLevel.INFO,
                        metric="lmm.output_tokens",
                        value=usage.completion_tokens,
                        unit="tokens",
                        attributes={"lmm.model": part.model},
                    )

                    if fingerprint := part.system_fingerprint:
                        ctx.log_debug(f"OpenAI system fingerprint:{fingerprint}")

                if part.choices:  # usage part does not contain choices
                    # we are always requesting single result - no need to take care of indices
                    element: Choice = part.choices[0]
                    # get the tool calls parts first
                    if tool_calls := element.delta.tool_calls:
                        # tool calls come in parts, we have to merge them manually
                        for call in tool_calls:
                            try:
                                tool_call: ChoiceDeltaToolCall = next(
                                    tool_call
                                    for tool_call in accumulated_tool_calls
                                    if tool_call.index == call.index
                                )

                                if call.id:
                                    if tool_call.id is not None:
                                        tool_call.id += call.id
                                    else:
                                        tool_call.id = call.id
                                else:
                                    pass

                                if call.function is None:
                                    continue

                                if tool_call.function is None:
                                    tool_call.function = call.function
                                    continue

                                if call.function.name:
                                    if tool_call.function.name is not None:
                                        tool_call.function.name += call.function.name
                                    else:
                                        tool_call.function.name = call.function.name
                                else:
                                    pass

                                if call.function.arguments:
                                    if tool_call.function.arguments is not None:
                                        tool_call.function.arguments += call.function.arguments
                                    else:
                                        tool_call.function.arguments = call.function.arguments
                                else:
                                    pass

                            except (StopIteration, StopAsyncIteration):
                                accumulated_tool_calls.append(call)

                    # then process content
                    if element.delta.content is not None:
                        content_chunk: LMMStreamChunk = LMMStreamChunk.of(
                            output_decoder(MultimodalContent.of(element.delta.content))
                        )
                        yield content_chunk

                    if finish_reason := element.finish_reason:
                        if finish_reason in ("length", "content_filter"):
                            raise OpenAIException(f"Unexpected finish reason: {finish_reason}")

                        if accumulated_tool_calls:
                            # send tool calls - openAI always sends it without other elements
                            for call in accumulated_tool_calls:
                                if not call.function:
                                    continue  # skip partial calls
                                if not call.function.name:
                                    continue  # skip calls with missing names

                                call_identifier: str = call.id or uuid4().hex

                                ctx.record(
                                    ObservabilityLevel.INFO,
                                    event="lmm.tool_request",
                                    attributes={"lmm.tool": call.function.name},
                                )

                                # send tool requests when ensured that all were completed
                                yield LMMToolRequest(
                                    identifier=call_identifier,
                                    tool=call.function.name,
                                    arguments=json.loads(call.function.arguments)
                                    if call.function.arguments
                                    else {},
                                )

                        else:
                            # send completion chunk - openAI sends it without an actual content
                            yield LMMStreamChunk.of(
                                MultimodalContent.empty,
                                eod=True,
                            )

        return ctx.stream(stream)
