from collections.abc import Mapping
from typing import Any, Literal, overload

from haiway import ctx

from draive.instructions.state import InstructionsRepository
from draive.instructions.types import Instruction, InstructionDeclaration, MissingInstruction

__all__ = [
    "fetch_instruction",
]


@overload
async def fetch_instruction(
    reference: InstructionDeclaration | str,
    /,
    *,
    arguments: Mapping[str, str] | None = None,
    **extra: Any,
) -> Instruction | None: ...


@overload
async def fetch_instruction(
    reference: InstructionDeclaration | str,
    /,
    *,
    default: Instruction | str,
    arguments: Mapping[str, str] | None = None,
    **extra: Any,
) -> Instruction: ...


@overload
async def fetch_instruction(
    reference: InstructionDeclaration | str,
    /,
    *,
    default: Instruction | str | None = None,
    arguments: Mapping[str, str] | None = None,
    required: Literal[True],
    **extra: Any,
) -> Instruction: ...


async def fetch_instruction(
    reference: InstructionDeclaration | str,
    /,
    *,
    default: Instruction | str | None = None,
    arguments: Mapping[str, str] | None = None,
    required: bool = True,
    **extra: Any,
) -> Instruction | None:
    name: str = reference if isinstance(reference, str) else reference.name

    match await ctx.state(InstructionsRepository).fetch(
        name,
        arguments=arguments,
        **extra,
    ):
        case None:
            match default:
                case None:
                    if required:
                        raise MissingInstruction(f"Missing instruction: '{name}'")

                    else:
                        return None

                case Instruction() as instruction:
                    return instruction

                case str() as text:
                    return Instruction.of(
                        text,
                        name=name,
                        description=reference.description
                        if isinstance(reference, InstructionDeclaration)
                        else None,
                        meta=None,
                        **(arguments if arguments is not None else {}),
                    )

        case instruction:
            return instruction
