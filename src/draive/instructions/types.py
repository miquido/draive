from collections.abc import Mapping
from typing import Any, Protocol, Self, final, overload, runtime_checkable
from uuid import uuid4

from haiway import State

__all__ = [
    "Instruction",
    "InstructionFetching",
    "MissingInstruction",
]


class MissingInstruction(Exception):
    pass


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
        **variables: str,
    ) -> str: ...

    @overload
    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **variables: str,
    ) -> str | None: ...

    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **variables: str,
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
        identifier: str | None = None,
        description: str | None = None,
        **variables: str,
    ) -> Self:
        match instruction:
            case str() as content:
                return cls(
                    identifier=identifier or uuid4().hex,
                    description=description,
                    content=content,
                    variables=variables,
                )

            case instruction:
                return instruction.updated(**variables)

    identifier: str
    description: str | None
    content: str
    variables: Mapping[str, str]

    def format(
        self,
        **variables: str,
    ) -> str:
        if variables:
            return self.content.format_map(
                {
                    **self.variables,
                    **variables,
                },
            )

        elif self.variables:
            return self.content.format_map(self.variables)

        else:
            return self.content

    def extended(
        self,
        instruction: str,
        /,
        description: str | None = None,
        joiner: str | None = None,
        **variables: str,
    ) -> Self:
        if variables:
            return self.__class__(
                identifier=self.identifier,
                description=description,
                content=(joiner or " ").join((self.content, instruction)),
                variables={
                    **self.variables,
                    **variables,
                },
            )

        else:
            return self.__class__(
                identifier=self.identifier,
                description=description,
                content=(joiner or " ").join((self.content, instruction)),
                variables=self.variables,
            )

    def updated(
        self,
        **variables: str,
    ) -> Self:
        if variables:
            return self.__class__(
                identifier=self.identifier,
                description=self.description,
                content=self.content,
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
            return self.content


@runtime_checkable
class InstructionFetching(Protocol):
    async def __call__(
        self,
        identifier: str,
        /,
        *,
        variables: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Instruction | None: ...
