from typing import Any

from haiway import ctx

from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps.state import Steps
from draive.steps.types import Step

__all__ = [
    "steps_completion",
]


async def steps_completion(
    *steps: Step | Multimodal,
    instruction: Instruction | str | None = None,
    **extra: Any,
) -> MultimodalContent:
    return await ctx.state(Steps).completion(
        *steps,
        instruction=instruction,
        **extra,
    )
