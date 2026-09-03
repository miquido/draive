from collections.abc import AsyncGenerator, MutableSequence, Sequence
from typing import Any

from haiway import as_list, ctx

from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationAssistantTurn,
    ConversationEvent,
    ConversationOutputStream,
    ConversationUserTurn,
)
from draive.models import (
    GenerativeModel,
    ModelContextElement,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.models.types import ModelOutputStream
from draive.multimodal import Multimodal, MultimodalContent, MultimodalContentPart
from draive.tools import Toolbox
from draive.utils import ProcessingEvent

__all__ = ("conversation_completion",)


async def conversation_completion(  # noqa: C901, PLR0912, PLR0915
    *,
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ConversationMemory,
    message: Multimodal,
    **extra: Any,
) -> ConversationOutputStream:
    model_context: MutableSequence[ModelContextElement] = as_list(await memory.recall())
    message = MultimodalContent.of(message)
    model_context.append(ModelInput.of(message))
    assistant_turn_accumulator: MutableSequence[
        MultimodalContent | ModelReasoning | ConversationEvent
    ] = []

    iteration: int = 0
    while True:  # loop until we get ModelOutput without tools
        async with ctx.scope(f"conversation.loop_{iteration}"):
            content_accumulator: MutableSequence[MultimodalContentPart] = []
            reasoning_accumulator: MutableSequence[ModelReasoningChunk] = []
            output_accumulator: MutableSequence[ModelOutputBlock] = []
            tool_requests: MutableSequence[ModelToolRequest] = []

            completion_stream: ModelOutputStream = GenerativeModel.completion(
                instructions=instructions,
                tools=toolbox.model_tools(iteration=iteration),
                context=model_context,
                output="auto",
                **extra,
            )
            try:
                async for chunk in completion_stream:
                    if isinstance(chunk, ModelReasoningChunk):
                        yield chunk

                        if content_accumulator:
                            content: MultimodalContent = MultimodalContent.of(*content_accumulator)
                            output_accumulator.append(content)
                            assistant_turn_accumulator.append(content)
                            content_accumulator.clear()

                        reasoning_accumulator.append(chunk)

                    elif isinstance(chunk, ModelToolRequest):  # TODO: start handling immediately
                        event: ConversationEvent = ConversationEvent.tool_request(chunk)
                        yield event

                        if content_accumulator:
                            content: MultimodalContent = MultimodalContent.of(*content_accumulator)
                            output_accumulator.append(content)
                            assistant_turn_accumulator.append(content)
                            content_accumulator.clear()

                        if reasoning_accumulator:
                            reasoning: Sequence[ModelReasoning] = ModelReasoning.blocks(
                                reasoning_accumulator
                            )
                            output_accumulator.extend(reasoning)
                            assistant_turn_accumulator.extend(reasoning)
                            reasoning_accumulator.clear()

                        output_accumulator.append(chunk)
                        tool_requests.append(chunk)
                        assistant_turn_accumulator.append(event)

                    else:
                        yield chunk

                        if reasoning_accumulator:
                            reasoning: Sequence[ModelReasoning] = ModelReasoning.blocks(
                                reasoning_accumulator
                            )
                            output_accumulator.extend(reasoning)
                            assistant_turn_accumulator.extend(reasoning)
                            reasoning_accumulator.clear()

                        content_accumulator.append(chunk)

            finally:
                await completion_stream.aclose()

            if content_accumulator:
                content: MultimodalContent = MultimodalContent.of(*content_accumulator)
                output_accumulator.append(content)
                assistant_turn_accumulator.append(content)

            if reasoning_accumulator:
                reasoning: Sequence[ModelReasoning] = ModelReasoning.blocks(reasoning_accumulator)
                output_accumulator.extend(reasoning)
                assistant_turn_accumulator.extend(reasoning)

            model_context.append(ModelOutput.of(*output_accumulator))

            if not tool_requests:
                break  # end of loop

            responses: MutableSequence[ModelToolResponse] = []
            tools_output_accumulator: MutableSequence[MultimodalContentPart] = []
            tools_stream: AsyncGenerator[
                ModelToolResponse | ProcessingEvent | MultimodalContentPart
            ] = toolbox.handle(*tool_requests)
            try:
                async for chunk in tools_stream:
                    if isinstance(chunk, ModelToolResponse):
                        responses.append(chunk)
                        event: ConversationEvent = ConversationEvent.tool_response(chunk)
                        assistant_turn_accumulator.append(event)
                        yield event

                    elif isinstance(chunk, ProcessingEvent):
                        event: ConversationEvent = ConversationEvent.tool_event(chunk)
                        assistant_turn_accumulator.append(event)
                        yield event

                    else:
                        tools_output_accumulator.append(chunk)
                        yield chunk

            finally:
                await tools_stream.aclose()

            ctx.log_debug("...received tool responses...")

            if tools_output_accumulator:  # tools direct result
                ctx.log_debug("...tools generated output...")
                tools_output = MultimodalContent.of(*tools_output_accumulator)
                model_context.append(ModelInput.of(*responses))
                model_context.append(ModelOutput.of(tools_output))
                assistant_turn_accumulator.append(tools_output)
                break  # end of loop

            else:  # regular tools result
                model_context.append(ModelInput.of(*responses))
                iteration += 1  # continue next iteration

    await memory.remember(
        ConversationUserTurn.of(message),
        ConversationAssistantTurn.of(*assistant_turn_accumulator),
    )
