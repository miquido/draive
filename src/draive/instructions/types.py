from typing import Protocol, Self, final, overload, runtime_checkable
from uuid import UUID, uuid4

from haiway import State

__all__ = [
    "Instruction",
    "InstructionFetching",
]


@final
class Instruction(State):
    @overload
    @classmethod
    def formatted(
        cls,
        instruction: None,
        /,
    ) -> None: ...

    @overload
    @classmethod
    def formatted(
        cls,
        instruction: Self | str,
        /,
        **variables: object,
    ) -> str: ...

    @overload
    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **variables: object,
    ) -> str | None: ...

    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **variables: object,
    ) -> str | None:
        match instruction:
            case None:
                return None

            case str() as string:
                if variables:
                    return string.format_map(variables)

                else:
                    return string

            case instruction:
                return instruction.format(**variables)

    @classmethod
    def of(
        cls,
        instruction: Self | str,
        /,
        identifier: UUID | None = None,
        **variables: object,
    ) -> Self:
        match instruction:
            case str() as content:
                return cls(
                    instruction=content,
                    identifier=identifier or uuid4(),
                    variables=variables,
                )

            case instruction:
                return instruction.updated(**variables)

    instruction: str
    identifier: UUID
    variables: dict[str, object]

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
                instruction=(joiner or " ").join((self.instruction, instruction)),
                identifier=self.identifier,
                variables={
                    **self.variables,
                    **variables,
                },
            )

        else:
            return self.__class__(
                instruction=(joiner or " ").join((self.instruction, instruction)),
                identifier=self.identifier,
                variables=self.variables,
            )

    def updated(
        self,
        **variables: object,
    ) -> Self:
        if variables:
            return self.__class__(
                instruction=self.instruction,
                identifier=self.identifier,
                variables={
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


@runtime_checkable
class InstructionFetching(Protocol):
    async def __call__(
        self,
        key: str,
    ) -> Instruction: ...
