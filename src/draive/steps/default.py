from typing import Any

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponse,
    lmm_invoke,
)
from draive.lmm.toolbox import Toolbox
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

        context: list[LMMContextElement] = []
        for step in steps:
            current_step: Step
            match step:
                case Step() as step:
                    current_step = step

                case input_content:
                    current_step = Step.of(input_content)

            await _process_step(
                current_step,
                instruction=instruction,
                context=context,
                **extra,
            )

        match context[-1]:  # the last element of the context should be the result
            case LMMCompletion() as result:
                return result.content

            case _:
                raise RuntimeError("Invalid steps completion state!")


async def _process_step(
    step: Step,
    /,
    *,
    instruction: Instruction | str | None,
    context: list[LMMContextElement],
    **extra: Any,
) -> None:
    context.append(LMMInput.of(step.input))

    toolbox: Toolbox = step.toolbox

    recursion_level: int = 0
    while recursion_level <= toolbox.repeated_calls_limit:
        match await lmm_invoke(
            instruction=step.instruction or instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            **step.extra,
            **extra,
        ):
            case LMMCompletion() as completion:
                return context.append(completion)

            case LMMToolRequests() as tool_requests:
                responses: list[LMMToolResponse] = await toolbox.respond_all(tool_requests)

                if direct_content := [
                    response.content for response in responses if response.direct
                ]:
                    return context.append(LMMCompletion.of(MultimodalContent.of(*direct_content)))

                else:
                    context.extend([tool_requests, *responses])

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")
