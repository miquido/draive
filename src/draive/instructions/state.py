from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self, final, overload

from haiway import META_EMPTY, Meta, State, ctx

from draive.instructions.file import InstructionsFileStorage
from draive.instructions.types import (
    Instruction,
    InstructionDeclaration,
    InstructionFetching,
    InstructionListFetching,
    InstructionMissing,
)
from draive.instructions.volatile import InstructionsVolatileStorage

__all__ = ("Instructions",)


async def _empty_list(
    **extra: Any,
) -> Sequence[InstructionDeclaration]:
    return ()


async def _no_instruction(
    name: str,
    /,
    *,
    arguments: Mapping[str, str | float | int] | None = None,
    **extra: Any,
) -> Instruction | None:
    return None


@final
class Instructions(State):
    @classmethod
    def of(
        cls,
        instruction: Instruction,
        *instructions: Instruction,
    ) -> Self:
        storage = InstructionsVolatileStorage((instruction, *instructions))

        return cls(
            list_fetching=storage.fetch_list,
            fetching=storage.fetch_instruction,
        )

    @classmethod
    def file(
        cls,
        path: Path | str,
    ) -> Self:
        storage = InstructionsFileStorage(path=path)

        return cls(
            list_fetching=storage.fetch_list,
            fetching=storage.fetch_instruction,
        )

    @classmethod
    async def fetch_list(
        cls,
        **extra: Any,
    ) -> Sequence[InstructionDeclaration]:
        return await ctx.state(cls).list_fetching(**extra)

    @overload
    @classmethod
    async def fetch(
        cls,
        instruction: InstructionDeclaration | str,
        /,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        **extra: Any,
    ) -> Instruction | None: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        instruction: InstructionDeclaration | str,
        /,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        default: Instruction | str,
        **extra: Any,
    ) -> Instruction: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        instruction: InstructionDeclaration | str,
        /,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        required: Literal[True],
        **extra: Any,
    ) -> Instruction: ...

    @classmethod
    async def fetch(
        cls,
        instruction: InstructionDeclaration | str,
        /,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        default: Instruction | str | None = None,
        required: bool = False,
        **extra: Any,
    ) -> Instruction | None:
        name: str = instruction if isinstance(instruction, str) else instruction.name

        if fetched := await ctx.state(Instructions).fetching(
            name,
            arguments=arguments,
            **extra,
        ):
            return fetched

        elif required and default is None:
            raise InstructionMissing(f"Missing instruction: '{name}'")

        elif default is None:
            return None

        elif isinstance(default, str):
            return Instruction.of(
                default,
                name=name,
                arguments=arguments,
            )

        else:
            return Instruction.of(
                default,
                arguments=arguments,
            )

    list_fetching: InstructionListFetching = _empty_list
    fetching: InstructionFetching = _no_instruction
    meta: Meta = META_EMPTY
