from typing import Any

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
    Toolbox,
    lmm_invoke,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps.types import Step

__all__ = [
    "default_steps_completion",
]


async def default_steps_completion(
    *steps: Step | Multimodal,
    instruction: Instruction | str | None = None,
    **extra: Any,
) -> MultimodalContent:
    with ctx.scope("steps_completion"):
        assert steps, "Steps cannot be empty"  # nosec: B101
        result_parts: list[MultimodalContent] = []
        last_step_result: MultimodalContent | None = None
        context: list[LMMContextElement] = []
        volatile_context_range: range | None = None
        for step in steps:
            current_step: Step
            match step:
                case Step() as step:
                    current_step = step

                case input_content:
                    current_step = Step.of(input_content)

            if current_step.condition is not None and not current_step.condition():
                continue  # skip unavailable steps

            context_end_index: int = len(context)
            step_result: MultimodalContent = await _process_step(
                current_step,
                instruction=instruction,
                context=context,
                **extra,
            )

            # remove volatile parts of context after using it
            if volatile_context_range is not None:
                del context[volatile_context_range.start : volatile_context_range.stop]
                # adjust end index if removed stuff
                context_end_index -= volatile_context_range.stop - volatile_context_range.start
                volatile_context_range = None

            # mark new volatile parts of context if any to be removed
            if current_step.volatile:
                volatile_context_range = range(
                    context_end_index,
                    len(context),
                )

            # include step result as part of the final result if needed
            if current_step.extends_result:
                result_parts.append(step_result)
                last_step_result = None  # make sure we won't make duplicates

            else:
                last_step_result = step_result

        if last_step_result is not None:
            result_parts.append(last_step_result)

        return MultimodalContent.of(*result_parts)


async def _process_step(
    step: Step,
    /,
    *,
    instruction: Instruction | str | None,
    context: list[LMMContextElement],
    **extra: Any,
) -> MultimodalContent:
    context.append(LMMInput.of(step.input))

    if step.completion is not None:
        context.append(step.completion)
        return step.completion.content

    toolbox: Toolbox = step.toolbox

    recursion_level: int = 0
    context_end_index: int = len(context)
    while recursion_level <= toolbox.repeated_calls_limit:
        match await lmm_invoke(
            instruction=step.instruction or instruction,
            context=context,
            output=step.output,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            **step.extra,
            **extra,
        ):
            case LMMCompletion() as completion:
                del context[context_end_index:]  # remove tool calls from context
                if step.result_processing is None:
                    context.append(completion)
                    return completion.content

                else:
                    processed_content: MultimodalContent = await step.result_processing(
                        completion.content
                    )
                    context.append(LMMCompletion.of(processed_content))
                    return processed_content

            case LMMToolRequests() as tool_requests:
                responses: list[LMMToolResponse] = await toolbox.respond_all(tool_requests)

                if direct_results := [
                    response.content for response in responses if response.direct
                ]:
                    del context[context_end_index:]  # remove tool calls from context
                    direct_content: MultimodalContent = MultimodalContent.of(*direct_results)
                    if step.result_processing is None:
                        context.append(LMMCompletion.of(direct_content))
                        return direct_content

                    else:
                        processed_content: MultimodalContent = await step.result_processing(
                            direct_content
                        )
                        context.append(LMMCompletion.of(processed_content))
                        return processed_content

                else:
                    context.extend(
                        [
                            tool_requests,
                            LMMToolResponses(responses=responses),
                        ]
                    )

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")
