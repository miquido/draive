from typing import Any

from haiway import ctx

from draive.instructions.state import InstructionsRepository
from draive.instructions.types import Instruction

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
