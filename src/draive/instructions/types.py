from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, final, overload, runtime_checkable
from uuid import uuid4

from haiway import Default, State

from draive.commons import META_EMPTY, Meta
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
    meta: Meta = Default(META_EMPTY)


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
        **arguments: str,
    ) -> str: ...

    @overload
    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **arguments: str,
    ) -> str | None: ...

    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **arguments: str,
    ) -> str | None:
        match instruction:
            case None:
                return None

            case str() as string:
                if arguments:
                    return string.format_map(arguments)

                else:
                    return string

            case instruction:
                return instruction.format(**arguments)

    @classmethod
    def of(
        cls,
        instruction: Self | str,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        meta: Meta | None = None,
        **arguments: str,
    ) -> Self:
        match instruction:
            case str() as content:
                return cls(
                    name=name or uuid4().hex,
                    description=description,
                    content=content,
                    arguments=arguments,
                    meta=meta if meta is not None else META_EMPTY,
                )

            case instruction:
                return instruction.updated(**arguments)

    name: str
    description: str | None = None
    content: str
    arguments: Mapping[str, str] = Default(factory=dict)
    meta: Meta = Default(META_EMPTY)

    def format(
        self,
        **arguments: str,
    ) -> str:
        if arguments:
            return self.content.format_map(
                {
                    **self.arguments,
                    **arguments,
                },
            )

        elif self.arguments:
            return self.content.format_map(self.arguments)

        else:
            return self.content

    def extended(
        self,
        instruction: str,
        /,
        description: str | None = None,
        joiner: str | None = None,
        **arguments: str,
    ) -> Self:
        if arguments:
            return self.__class__(
                name=self.name,
                description=description,
                content=(joiner if joiner is not None else "").join((self.content, instruction)),
                arguments={
                    **self.arguments,
                    **arguments,
                },
                meta=self.meta,
            )

        else:
            return self.__class__(
                name=self.name,
                description=description,
                content=(joiner if joiner is not None else "").join((self.content, instruction)),
                arguments=self.arguments,
                meta=self.meta,
            )

    def with_arguments(
        self,
        **arguments: str,
    ) -> Self:
        if arguments:
            return self.__class__(
                name=self.name,
                description=self.description,
                content=self.content,
                arguments={
                    **self.arguments,
                    **arguments,
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
