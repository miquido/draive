from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any, Protocol, Self, final, overload, runtime_checkable
from uuid import uuid4

from haiway import Default, State

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.parameters import DataModel

__all__ = (
    "Instruction",
    "InstructionDeclaration",
    "InstructionDeclarationArgument",
    "InstructionException",
    "InstructionFetching",
    "InstructionListFetching",
    "InstructionMissing",
)


class InstructionException(Exception):
    pass


class InstructionMissing(InstructionException):
    pass


@final
class InstructionDeclarationArgument(DataModel):
    name: str
    description: str | None = None
    required: bool = True


@final
class InstructionDeclaration(DataModel):
    name: str
    description: str | None = None
    arguments: Sequence[InstructionDeclarationArgument]
    meta: Meta = META_EMPTY


@final
class Instruction(State):
    @overload
    @classmethod
    def formatted(
        cls,
        instruction: Self | str,
        /,
        **arguments: str | float | int,
    ) -> str: ...

    @overload
    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **arguments: str | float | int,
    ) -> str | None: ...

    @classmethod
    def formatted(
        cls,
        instruction: Self | str | None,
        /,
        **arguments: str | float | int,
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

    @overload
    @classmethod
    def of(
        cls,
        instruction: Self | str,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        meta: Meta | MetaValues | None = None,
        arguments: Mapping[str, str | float | int] | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        instruction: Self | str | None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        meta: Meta | MetaValues | None = None,
        arguments: Mapping[str, str | float | int] | None = None,
    ) -> Self | None: ...

    @classmethod
    def of(
        cls,
        instruction: Self | str | None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        meta: Meta | MetaValues | None = None,
        arguments: Mapping[str, str | float | int] | None = None,
    ) -> Self | None:
        match instruction:
            case None:
                return None

            case str() as content:
                return cls(
                    name=name or uuid4().hex,
                    description=description,
                    content=content,
                    arguments=arguments if arguments is not None else {},
                    meta=Meta.of(meta),
                )

            case instruction:
                if name is None and description is None and meta is None and arguments is None:
                    return instruction  # nothing to update

                return instruction.updated(
                    name=name or instruction.name,
                    description=description or instruction.description,
                    content=instruction.content,
                    arguments={**instruction.arguments, **arguments}
                    if arguments is not None
                    else instruction.arguments,
                    meta=instruction.meta.merged_with(meta)
                    if meta is not None
                    else instruction.meta,
                )

    name: str
    description: str | None = None
    content: str
    arguments: Mapping[str, str | float | int] = Default(factory=dict)
    meta: Meta = META_EMPTY

    @property
    def declaration(self) -> InstructionDeclaration:
        return InstructionDeclaration(
            name=self.name,
            description=self.description,
            arguments=extract_variables(self.content),
            meta=self.meta,
        )

    def format(
        self,
        **arguments: str | float | int,
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
        joiner: str | None = None,
        **arguments: str | float | int,
    ) -> Self:
        if arguments:
            return self.__class__(
                name=self.name,
                description=self.description,
                content=(joiner if joiner is not None else "").join((self.content, instruction))
                if instruction
                else self.content,
                arguments={
                    **self.arguments,
                    **arguments,
                },
                meta=self.meta,
            )

        elif instruction:
            return self.__class__(
                name=self.name,
                description=self.description,
                content=(joiner if joiner is not None else "").join((self.content, instruction)),
                arguments=self.arguments,
                meta=self.meta,
            )

        else:  # nothing to update
            return self


def extract_variables(
    string: str,
    /,
) -> Sequence[InstructionDeclarationArgument]:
    seen: set[str] = set()
    unique_args: list[InstructionDeclarationArgument] = []
    for _, field_name, _, _ in Formatter().parse(string):
        if field_name and field_name not in seen:
            seen.add(field_name)
            unique_args.append(InstructionDeclarationArgument(name=field_name))

    return unique_args


@runtime_checkable
class InstructionFetching(Protocol):
    async def __call__(
        self,
        name: str,
        /,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        **extra: Any,
    ) -> Instruction | None: ...


@runtime_checkable
class InstructionListFetching(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[InstructionDeclaration]: ...
