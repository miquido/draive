import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from itertools import chain
from typing import Any, Literal, cast, overload
from uuid import uuid4

from haiway import ArgumentsTrace, ResultTrace, as_list, ctx
from mistralai import (
    CompletionEvent,
    CompletionResponseStreamChoice,
    DeltaMessage,
    ToolCall,
    ToolTypedDict,
)
from mistralai.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequestToolChoiceTypedDict,
    ChatCompletionResponse,
    MessagesTypedDict,
    ResponseFormatTypedDict,
    StopTypedDict,
)
from mistralai.types.basemodel import Unset
from mistralai.utils.eventstreaming import EventStreamAsync

from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMOutput,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.lmm.types import LMMStreamChunk, LMMStreamOutput
from draive.metrics import TokenUsage
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralChatConfig
from draive.mistral.lmm import (
    content_chunk_as_content_element,
    content_element_as_content_chunk,
    context_element_as_messages,
    output_as_response_declaration,
    tools_as_tool_config,
)
from draive.mistral.types import MistralException
from draive.mistral.utils import unwrap_missing_to_none, unwrap_missing_to_unset
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("MistralLMMGeneration",)


class MistralLMMGeneration(MistralAPI):
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
        config: MistralChatConfig | None = None,
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
        config: MistralChatConfig | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        prefill: Multimodal | None = None,
        config: MistralChatConfig | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        with ctx.scope("mistral_lmm_completion"):
            completion_config: MistralChatConfig = config or ctx.state(MistralChatConfig).updated(
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

            messages: list[MessagesTypedDict] = list(
                chain.from_iterable([context_element_as_messages(element) for element in context])
            )

            if prefill:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            content_element_as_content_chunk(element)
                            for element in MultimodalContent.of(prefill).parts
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

            response_format, output_decoder = output_as_response_declaration(output)

            tool_choice, tools_list = tools_as_tool_config(
                tools,
                tool_selection=tool_selection,
            )

            if stream:
                return await self._completion_stream(
                    model=completion_config.model,
                    messages=messages,
                    temperature=completion_config.temperature,
                    top_p=unwrap_missing_to_none(completion_config.top_p),
                    max_tokens=unwrap_missing_to_unset(completion_config.max_tokens),
                    stop=as_list(unwrap_missing_to_none(completion_config.stop_sequences)),
                    random_seed=unwrap_missing_to_unset(completion_config.seed),
                    response_format=response_format,
                    tools=tools_list,
                    tool_choice=tool_choice,
                    output_decoder=output_decoder,
                )

            else:
                return await self._completion(
                    model=completion_config.model,
                    messages=messages,
                    temperature=completion_config.temperature,
                    top_p=unwrap_missing_to_none(completion_config.top_p),
                    max_tokens=unwrap_missing_to_unset(completion_config.max_tokens),
                    stop=as_list(unwrap_missing_to_none(completion_config.stop_sequences)),
                    random_seed=unwrap_missing_to_unset(completion_config.seed),
                    response_format=response_format,
                    tools=tools_list,
                    tool_choice=tool_choice,
                    output_decoder=output_decoder,
                )

    async def _completion(
        self,
        model: str,
        messages: list[MessagesTypedDict],
        temperature: float,
        top_p: float | None,
        max_tokens: int | Unset,
        stop: StopTypedDict | None,
        random_seed: int | Unset,
        response_format: ResponseFormatTypedDict,
        tools: list[ToolTypedDict] | Unset,
        tool_choice: ChatCompletionRequestToolChoiceTypedDict | None,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> LMMOutput:
        completion: ChatCompletionResponse = await self._client.chat.complete_async(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            random_seed=random_seed,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )

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
                    lmm_completion = LMMCompletion.of(output_decoder(MultimodalContent.of(string)))

                case chunks:
                    lmm_completion = LMMCompletion.of(
                        output_decoder(
                            MultimodalContent.of(
                                *[content_chunk_as_content_element(chunk) for chunk in chunks]
                            )
                        ),
                    )

        else:
            lmm_completion = None

        if tool_calls := completion_message.tool_calls:
            completion_tool_calls = LMMToolRequests(
                content=lmm_completion.content if lmm_completion else None,
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

    async def _completion_stream(  # noqa: C901, PLR0915
        self,
        model: str,
        messages: list[MessagesTypedDict],
        temperature: float,
        top_p: float | None,
        max_tokens: int | Unset,
        stop: StopTypedDict | None,
        random_seed: int | Unset,
        response_format: ResponseFormatTypedDict,
        tools: list[ToolTypedDict] | Unset,
        tool_choice: ChatCompletionRequestToolChoiceTypedDict | None,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> AsyncIterator[LMMStreamOutput]:
        response_stream: EventStreamAsync[CompletionEvent] = await self._client.chat.stream_async(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            random_seed=random_seed,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        async def stream() -> AsyncGenerator[LMMStreamOutput]:  # noqa: C901, PLR0912, PLR0915
            async with response_stream:
                accumulated_tool_calls: list[ToolCall] = []
                async for completion_chunk in response_stream:
                    if usage := completion_chunk.data.usage:
                        ctx.record(
                            TokenUsage.for_model(
                                completion_chunk.data.model,
                                input_tokens=usage.prompt_tokens,
                                cached_tokens=None,
                                output_tokens=usage.completion_tokens,
                            ),
                        )

                    if not completion_chunk.data.choices:
                        raise MistralException(
                            "Invalid Mistral completion - missing choices!",
                            completion_chunk.data,
                        )

                    completion_choice: CompletionResponseStreamChoice = (
                        completion_chunk.data.choices[0]
                    )

                    completion_delta: DeltaMessage = completion_choice.delta
                    if content := completion_delta.content:
                        match content:
                            case str() as string:
                                yield LMMStreamChunk.of(
                                    output_decoder(MultimodalContent.of(string))
                                )

                            case chunks:
                                yield LMMStreamChunk.of(
                                    output_decoder(
                                        MultimodalContent.of(
                                            *[
                                                content_chunk_as_content_element(chunk)
                                                for chunk in chunks
                                            ]
                                        )
                                    )
                                )

                    if tool_calls := completion_delta.tool_calls:
                        if not accumulated_tool_calls:
                            accumulated_tool_calls = sorted(
                                tool_calls,
                                key=lambda call: call.index or 0,
                            )

                        else:
                            for tool_call in tool_calls:
                                assert tool_call.index, "Can't identify function call without index"  # nosec: B101

                                # "null" is a dafault value...
                                if tool_call.id and tool_call.id != "null":
                                    accumulated_tool_calls[tool_call.index].id = tool_call.id

                                if tool_call.function.name:
                                    accumulated_tool_calls[
                                        tool_call.index
                                    ].function.name += tool_call.function.name

                                if isinstance(tool_call.function.arguments, str):
                                    assert isinstance(  # nosec: B101
                                        accumulated_tool_calls[tool_call.index].function.arguments,
                                        str,
                                    )
                                    accumulated_tool_calls[  # pyright: ignore[reportOperatorIssue]
                                        tool_call.index
                                    ].function.arguments += tool_call.function.arguments

                                else:
                                    assert isinstance(  # nosec: B101
                                        accumulated_tool_calls[tool_call.index].function.arguments,
                                        dict,
                                    )
                                    accumulated_tool_calls[tool_call.index].function.arguments = {
                                        **cast(
                                            dict,
                                            accumulated_tool_calls[
                                                tool_call.index
                                            ].function.arguments,
                                        ),
                                        **tool_call.function.arguments,
                                    }

                    match completion_choice.finish_reason:
                        case None:
                            pass  # continue streaming

                        case "stop":
                            # send tool calls if any
                            for call in accumulated_tool_calls:
                                if not call.function:
                                    continue  # skip partial calls
                                if not call.function.name:
                                    continue  # skip calls with missing names

                                call_identifier: str = call.id or uuid4().hex
                                # send tool requests when ensured that all were completed
                                yield LMMToolRequest(
                                    identifier=call_identifier,
                                    tool=call.function.name,
                                    arguments=json.loads(call.function.arguments)
                                    if isinstance(call.function.arguments, str)
                                    else call.function.arguments,
                                )
                            # send completion chunk if needed
                            yield LMMStreamChunk.of(
                                MultimodalContent.empty,
                                eod=True,
                            )
                            break  # and break the loop

                        case "tool_calls":
                            # send tool calls if any
                            for call in accumulated_tool_calls:
                                if not call.function:
                                    continue  # skip partial calls
                                if not call.function.name:
                                    continue  # skip calls with missing names

                                call_identifier: str = call.id or uuid4().hex
                                # send tool requests when ensured that all were completed
                                yield LMMToolRequest(
                                    identifier=call_identifier,
                                    tool=call.function.name,
                                    arguments=json.loads(call.function.arguments)
                                    if isinstance(call.function.arguments, str)
                                    else call.function.arguments,
                                )

                            break  # and break the loop

                        case "length":
                            raise MistralException(
                                "Invalid Mistral completion - exceeded maximum length!",
                                completion_chunk.data,
                            )

                        case "error":
                            raise MistralException(
                                "Mistral completion generation failed!",
                                completion_chunk.data,
                            )

        return ctx.stream(stream)
