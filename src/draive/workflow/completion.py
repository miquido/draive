from collections.abc import MutableSequence
from typing import Any

from draive.lmm.types import LMMContextElement
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.workflow.stage import Stage

__all__ = [
    "workflow_completion",
]


async def workflow_completion(
    stage: Stage | Prompt | Multimodal,
    /,
    *stages: Stage | Prompt | Multimodal,
    **extra: Any,
) -> MultimodalContent:
    context: MutableSequence[LMMContextElement] = []
    current_result = MultimodalContent.of()
    for current in (stage, *stages):
        current_stage: Stage
        match current:
            case Stage() as stage:
                current_stage = stage

            case Prompt() as prompt:
                current_stage = Stage.completion(prompt)

            case multimodal:
                current_stage = Stage.completion(multimodal)

        current_result = await current_stage._processing(
            context=context,
            result=current_result,
        )

    return current_result
