from collections.abc import Mapping, Sequence
from typing import Any, final

from draive.instructions.types import (
    Instruction,
    InstructionDeclaration,
    InstructionMissing,
)

__all__ = ("InstructionsVolatileStorage",)


@final
class InstructionsVolatileStorage:
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

    async def listing(
        self,
        **extra: Any,
    ) -> Sequence[InstructionDeclaration]:
        return self._listing

    async def instruction(
        self,
        name: str,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        **extra: Any,
    ) -> Instruction:
        if instruction := self._storage.get(name):
            if arguments:
                return instruction.updated(arguments={**instruction.arguments, **arguments})

            else:
                return instruction

        else:
            raise InstructionMissing(f"Instruction '{name}' is not defined")
