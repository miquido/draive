from typing import Self, final
from uuid import uuid4

from draive.utils import freeze

__all__ = [
    "Instruction",
]


@final
class Instruction:
    @classmethod
    def of(
        cls,
        instruction: Self | str,
        /,
        **variables: object,
    ) -> Self:
        match instruction:
            case str() as content:
                return cls(
                    content,
                    identifier=None,
                    **variables,
                )

            case instruction:
                return instruction.updated(**variables)

    def __init__(
        self,
        instruction: str,
        /,
        identifier: str | None = None,
        **variables: object,
    ) -> None:
        self.instruction: str = instruction
        self.identifier: str = identifier or uuid4().hex
        self.variables: dict[str, object] = variables

        freeze(self)

    def format(
        self,
        **variables: object,
    ) -> str:
        if variables:
            return self.instruction.format_map(
                {
                    **self.variables,
                    **variables,
                },
            )

        elif self.variables:
            return self.instruction.format_map(self.variables)

        else:
            return self.instruction

    def extended(
        self,
        instruction: str,
        /,
        joiner: str | None = None,
        **variables: object,
    ) -> Self:
        if variables:
            return self.__class__(
                (joiner or " ").join((self.instruction, instruction)),
                identifier=self.identifier,
                **{
                    **self.variables,
                    **variables,
                },
            )

        else:
            return self.__class__(
                (joiner or " ").join((self.instruction, instruction)),
                identifier=self.identifier,
                **self.variables,
            )

    def updated(
        self,
        **variables: object,
    ) -> Self:
        if variables:
            return self.__class__(
                self.instruction,
                identifier=self.identifier,
                **{
                    **self.variables,
                    **variables,
                },
            )

        else:
            return self

    def __str__(self) -> str:
        try:
            return self.format()

        except KeyError:
            return self.instruction
