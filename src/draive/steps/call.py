from collections.abc import Iterable
from typing import Any

from draive.instructions import Instruction
from draive.scope import ctx
from draive.steps.model import Step
from draive.steps.state import Steps
from draive.types import Multimodal, MultimodalContent

__all__ = [
    "steps_completion",
]


async def steps_completion(
    *,
    instruction: Instruction | str,
    steps: Iterable[Step | Multimodal],
    **extra: Any,
) -> MultimodalContent:
    return await ctx.state(Steps).completion(
        instruction=instruction,
        steps=steps,
        **extra,
    )
