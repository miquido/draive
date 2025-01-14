from haiway import State, async_always

from draive.instructions.types import InstructionFetching

__all__ = [
    "InstructionsRepository",
]


class InstructionsRepository(State):
    fetch: InstructionFetching = async_always(None)
