from collections.abc import Mapping
from typing import Any, Literal, overload

from haiway import ctx

from draive.instructions.state import InstructionsRepository
from draive.instructions.types import Instruction, MissingInstruction

__all__ = [
    "instruction",
]


@overload
async def instruction(
    key: str,
    /,
    *,
    default: Instruction | str | None = None,
    variables: Mapping[str, str] | None = None,
    **extra: Any,
) -> Instruction | None: ...


@overload
async def instruction(
    key: str,
    /,
    *,
    default: Instruction | str | None = None,
    variables: Mapping[str, str] | None = None,
    required: Literal[True],
    **extra: Any,
) -> Instruction | None: ...


async def instruction(
    key: str,
    /,
    *,
    default: Instruction | str | None = None,
    variables: Mapping[str, str] | None = None,
    required: bool = True,
    **extra: Any,
) -> Instruction | None:
    match await ctx.state(InstructionsRepository).fetch(
        key,
        variables=variables,
        **extra,
    ):
        case None:
            if default is not None:
                return Instruction.of(
                    default,
                    identifier=None,
                    **(variables if variables is not None else {}),
                )

            elif required:
                raise MissingInstruction(f"Missing instruction: '{key}'")

            else:
                return None

        case instruction:
            return instruction
