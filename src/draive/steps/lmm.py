from collections.abc import Iterable
from typing import Any

from draive.instructions import Instruction
from draive.lmm import lmm_invocation
from draive.lmm.tools.toolbox import Toolbox
from draive.scope import ctx
from draive.steps.model import Step
from draive.types import (
    LMMContextElement,
    LMMInput,
    Multimodal,
    MultimodalContent,
)
from draive.types.lmm import LMMCompletion, LMMToolRequests, LMMToolResponse

__all__ = [
    "lmm_steps_completion",
]


async def lmm_steps_completion(
    *,
    instruction: Instruction | str,
    steps: Iterable[Step | Multimodal],
    **extra: Any,
) -> MultimodalContent:
    with ctx.nested(
        "lmm_steps_completion",
    ):
        assert steps, "Steps cannot be empty"  # nosec: B101

        context: list[LMMContextElement] = []
        for step in steps:
            current_step: Step
            match step:
                case Step() as step:
                    current_step = step

                case input_content:
                    current_step = Step.of(input_content)

            await _lmm_process_step(
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


async def _lmm_process_step(
    step: Step,
    /,
    instruction: Instruction | str,
    context: list[LMMContextElement],
    **extra: Any,
) -> None:
    context.append(LMMInput.of(step.input))

    if prefill := step.prefill:
        context.append(LMMCompletion.of(prefill))

    toolbox: Toolbox = step.toolbox

    recursion_level: int = 0
    while recursion_level <= toolbox.recursion_limit:
        match await lmm_invocation(
            instruction=step.instruction or instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(recursion_level=recursion_level),
            output="text",
            stream=False,
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("Received step result")
                return context.append(completion)

            case LMMToolRequests() as tool_requests:
                ctx.log_debug("Received step tool calls")
                responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

                if direct_content := [
                    response.content for response in responses if response.direct
                ]:
                    return context.append(LMMCompletion.of(MultimodalContent.of(*direct_content)))

                elif prefill := step.prefill:  # move prefill to the next completion if used tools
                    del context[-1]
                    context.extend([tool_requests, *responses, LMMCompletion.of(prefill)])

                else:
                    context.extend([tool_requests, *responses])

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")
