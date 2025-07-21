from collections.abc import Mapping, Sequence
from typing import Any, final

from haiway import ctx

from draive.instructions.types import (
    Instruction,
    InstructionDeclaration,
)

__all__ = ("InstructionsVolatileStorage",)


@final
class InstructionsVolatileStorage:
    __slots__ = (
        "_listing",
        "_storage",
    )

    def __init__(
        self,
        instructions: Sequence[Instruction],
    ) -> None:
        self._storage: Mapping[str, Instruction] = {
            instruction.name: instruction for instruction in instructions
        }
        self._listing: Sequence[InstructionDeclaration] = [
            instruction.declaration for instruction in instructions
        ]

    async def fetch_list(
        self,
        **extra: Any,
    ) -> Sequence[InstructionDeclaration]:
        return self._listing

    async def fetch_instruction(
        self,
        name: str,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        **extra: Any,
    ) -> Instruction | None:
        if instruction := self._storage.get(name):
            if arguments:
                return instruction.updated(arguments={**instruction.arguments, **arguments})

            else:
                return instruction

        else:
            ctx.log_debug(f"Instruction '{name}' is not defined")
            return None
