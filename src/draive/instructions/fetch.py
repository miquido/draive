from typing import Any

from draive.instructions.state import InstructionsRepository
from draive.instructions.types import Instruction
from draive.scope import ctx

__all__ = [
    "instruction",
]


async def instruction(
    key: str,
    /,
    *,
    default: Instruction | str | None = None,
    **extra: Any,
) -> Instruction:
    return (
        await ctx.state(InstructionsRepository)
        .updated(**extra)
        .instruction(
            key,
            default=default,
        )
    )
