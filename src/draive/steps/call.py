from typing import Any

from haiway import ctx
from typing_extensions import deprecated

from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.steps.state import Steps
from draive.steps.types import Step

__all__ = [
    "steps_completion",
]


@deprecated("`steps_completion` has been replaced with `Stage`")
async def steps_completion(
    *steps: Step | Prompt | Multimodal,
    instruction: Instruction | str | None = None,
    **extra: Any,
) -> MultimodalContent:
    return await ctx.state(Steps).completion(
        *steps,
        instruction=instruction,
        **extra,
    )
