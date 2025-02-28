from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, final, overload, runtime_checkable
from uuid import uuid4

from haiway import State

from draive.parameters import DataModel, Field, ParameterSpecification

__all__ = [
    "Instruction",
    "InstructionDeclaration",
    "InstructionDeclarationArgument",
    "InstructionFetching",
    "MissingInstruction",
]


class MissingInstruction(Exception):
    pass


class InstructionDeclarationArgument(DataModel):
    name: str
    specification: ParameterSpecification = Field(
        specification={
            "type": "object",
            "additionalProperties": True,
        }
    )
    required: bool = True


class InstructionDeclaration(DataModel):
    name: str
    description: str | None = None
    arguments: Sequence[InstructionDeclarationArgument]
    meta: Mapping[str, str | float | int | bool | None] | None


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
        *,
        name: str | None = None,
        description: str | None = None,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
        **variables: str,
    ) -> Self:
        match instruction:
            case str() as content:
                return cls(
                    name=name or uuid4().hex,
                    description=description,
                    content=content,
                    variables=variables,
                    meta=meta,
                )

            case instruction:
                return instruction.updated(**variables)

    name: str
    description: str | None
    content: str
    variables: Mapping[str, str]
    meta: Mapping[str, str | float | int | bool | None] | None

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
                name=self.name,
                description=description,
                content=(joiner if joiner is not None else "").join((self.content, instruction)),
                variables={
                    **self.variables,
                    **variables,
                },
                meta=self.meta,
            )

        else:
            return self.__class__(
                name=self.name,
                description=description,
                content=(joiner if joiner is not None else "").join((self.content, instruction)),
                variables=self.variables,
                meta=self.meta,
            )

    def updated(
        self,
        **variables: str,
    ) -> Self:
        if variables:
            return self.__class__(
                name=self.name,
                description=self.description,
                content=self.content,
                variables={
                    **self.variables,
                    **variables,
                },
                meta=self.meta,
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
        name: str,
        /,
        *,
        arguments: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Instruction | None: ...
