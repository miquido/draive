from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, final, runtime_checkable

from haiway import META_EMPTY, Meta, MetaValues, State

__all__ = (
    "Instructions",
    "InstructionsArgumentDeclaration",
    "InstructionsDeclaration",
    "InstructionsDefining",
    "InstructionsListing",
    "InstructionsLoading",
    "InstructionsMissing",
    "InstructionsRemoving",
)


class InstructionsMissing(Exception):
    """Raised when a named instruction cannot be found in the repository."""

    __slots__ = ("name",)

    def __init__(
        self,
        *,
        name: str,
    ) -> None:
        super().__init__(f"Missing instruction - {name}")
        self.name: str = name


@final
class Instructions(State):
    """Reference to named instructions with optional arguments and metadata.

    Used to defer resolving of instruction text until execution time against an
    ``InstructionsRepository``.

    Attributes
    ----------
    name : str
        Instruction name or key.
    arguments : Mapping[str, str]
        Template variables for formatting the instructions content.
    meta : Meta
        Additional metadata for the lookup.
    """

    @classmethod
    def of(
        cls,
        name: str,
        /,
        *,
        arguments: Mapping[str, str] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an instructions reference.

        Parameters
        ----------
        name : str
            Name of the instructions to resolve.
        arguments : Mapping[str, str] | None, optional
            Variables used to format the resolved content.
        meta : Meta | Mapping[str, Any] | None, optional
            Additional metadata for repository backends.
        """
        return cls(
            name=name,
            arguments=arguments if arguments is not None else {},
            meta=Meta.of(meta),
        )

    name: str
    arguments: Mapping[str, str]
    meta: Meta

    def with_arguments(
        self,
        **arguments: str,
    ) -> Self:
        """Return a new reference with additional/overridden arguments."""
        if not arguments:
            return self

        return self.__class__(
            name=self.name,
            arguments={**self.arguments, **arguments},
            meta=self.meta,
        )

    def with_meta(
        self,
        meta: Meta | MetaValues,
        /,
    ) -> Self:
        """Return a new reference with merged metadata."""
        if not meta:
            return self

        return self.__class__(
            name=self.name,
            arguments=self.arguments,
            meta=self.meta.merged_with(meta),
        )


@final
class InstructionsArgumentDeclaration(State):
    """Describes a single argument for an instructions template."""

    @classmethod
    def of(
        cls,
        name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> Self:
        """Create an argument declaration."""
        return cls(
            name=name,
            description=description,
            required=required,
        )

    name: str
    description: str | None = None
    required: bool = True


@final
class InstructionsDeclaration(State):
    """Describes a named instructions template.

    Attributes
    ----------
    name : str
        Template name/key.
    arguments : Sequence[InstructionsArgumentDeclaration]
        Declared arguments for the template.
    description : str | None
        Human-readable description.
    meta : Meta
        Additional metadata for storage/backends.
    """

    @classmethod
    def of(
        cls,
        name: str,
        /,
        *,
        arguments: Sequence[InstructionsArgumentDeclaration] = (),
        description: str | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a declaration for a named instructions template."""
        return cls(
            name=name,
            arguments=arguments,
            description=description,
            meta=Meta.of(meta),
        )

    name: str
    arguments: Sequence[InstructionsArgumentDeclaration] = ()
    description: str | None = None
    meta: Meta = META_EMPTY

    def to_instructions(
        self,
        *,
        arguments: Mapping[str, str] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Instructions:
        """Instantiate this template into an ``Instructions`` reference."""
        # should we validate required arguments?
        return Instructions.of(
            self.name,
            arguments=arguments if arguments is not None else {},
            meta=self.meta.merged_with(meta),
        )


@runtime_checkable
class InstructionsListing(Protocol):
    """Callable returning available instructions declarations."""

    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[InstructionsDeclaration]: ...


@runtime_checkable
class InstructionsLoading(Protocol):
    """Callable that loads the content of a named instructions template."""

    async def __call__(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None: ...


@runtime_checkable
class InstructionsDefining(Protocol):
    """Callable that defines or updates an instructions template content."""

    async def __call__(
        self,
        declaration: InstructionsDeclaration,
        content: str,
        **extra: Any,
    ) -> None: ...


@runtime_checkable
class InstructionsRemoving(Protocol):
    """Callable that removes a named instructions template."""

    async def __call__(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> None: ...
