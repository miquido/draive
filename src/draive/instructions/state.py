from collections.abc import Mapping
from typing import Any

from haiway import State

from draive.instructions.types import Instruction, InstructionFetching

__all__ = [
    "InstructionsRepository",
]


async def _empty_repository(
    name: str,
    /,
    *,
    arguments: Mapping[str, str] | None = None,
    **extra: Any,
) -> Instruction | None:
    return None


class InstructionsRepository(State):
    fetch: InstructionFetching = _empty_repository
