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
    arguments : Mapping[str, str | int | float]
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
        arguments: Mapping[str, str | int | float] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an instructions reference.

        Parameters
        ----------
        name : str
            Name of the instructions to resolve.
        arguments : Mapping[str, str | int | float] | None, optional
            Variables used to format the resolved content.
        meta : Meta | MetaValues | None, optional
            Additional metadata for repository backends.
        """
        return cls(
            name=name,
            arguments=arguments if arguments is not None else {},
            meta=Meta.of(meta),
        )

    name: str
    arguments: Mapping[str, str | int | float]
    meta: Meta

    def with_arguments(
        self,
        **arguments: str | int | float,
    ) -> Self:
        """Return a new reference with additional or overridden arguments.

        Parameters
        ----------
        **arguments : str | int | float
            Template variables merged into the existing arguments mapping.

        Returns
        -------
        Instructions
            New reference carrying the updated arguments.
        """
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
        """Return a new reference with merged metadata.

        Parameters
        ----------
        meta : Meta | MetaValues
            Additional metadata merged with the existing reference metadata.

        Returns
        -------
        Instructions
            New reference containing the merged metadata.
        """
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
        """Create an argument declaration.

        Parameters
        ----------
        name : str
            Argument name exposed to templates.
        description : str | None, optional
            Optional human-readable description.
        required : bool, optional
            Flag indicating whether the argument is mandatory.

        Returns
        -------
        InstructionsArgumentDeclaration
            New argument declaration instance.
        """
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
        """Create a declaration for a named instructions template.

        Parameters
        ----------
        name : str
            Template identifier.
        arguments : Sequence[InstructionsArgumentDeclaration], optional
            Declared arguments accepted by the template.
        description : str | None, optional
            Optional human-readable description.
        meta : Meta | MetaValues | None, optional
            Metadata associated with the template.

        Returns
        -------
        InstructionsDeclaration
            Declaration representing the template.
        """
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
        """Instantiate this template into an ``Instructions`` reference.

        Parameters
        ----------
        arguments : Mapping[str, str] | None, optional
            Arguments used when formatting the template.
        meta : Meta | MetaValues | None, optional
            Additional metadata merged with the declaration metadata.

        Returns
        -------
        Instructions
            Reference to the template ready for resolution.
        """
        # should we validate required arguments?
        return Instructions.of(
            self.name,
            arguments=arguments if arguments is not None else {},
            meta=self.meta.merged_with(meta),
        )


@runtime_checkable
class InstructionsListing(Protocol):
    """Callable returning available instructions declarations.

    Parameters
    ----------
    **extra : Any
        Extra keyword arguments forwarded to the implementation.

    Returns
    -------
    Sequence[InstructionsDeclaration]
        Available declarations discovered by the backend.
    """

    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[InstructionsDeclaration]: ...


@runtime_checkable
class InstructionsLoading(Protocol):
    """Callable that loads the content of a named instructions template.

    Parameters
    ----------
    name : str
        Template identifier to load.
    meta : Meta
        Metadata associated with the template lookup.
    **extra : Any
        Extra keyword arguments forwarded to the implementation.

    Returns
    -------
    str | None
        Loaded content or ``None`` when not available.
    """

    async def __call__(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None: ...


@runtime_checkable
class InstructionsDefining(Protocol):
    """Callable that defines or updates an instructions template content.

    Parameters
    ----------
    declaration : InstructionsDeclaration
        Template declaration describing what to define.
    content : str
        Template body stored in the backend.
    **extra : Any
        Extra keyword arguments forwarded to the implementation.

    Returns
    -------
    None
        Implementations perform side effects only.
    """

    async def __call__(
        self,
        declaration: InstructionsDeclaration,
        content: str,
        **extra: Any,
    ) -> None: ...


@runtime_checkable
class InstructionsRemoving(Protocol):
    """Callable that removes a named instructions template.

    Parameters
    ----------
    name : str
        Template identifier to remove.
    meta : Meta
        Metadata aiding the removal operation.
    **extra : Any
        Extra keyword arguments forwarded to the implementation.

    Returns
    -------
    None
        Implementations perform side effects only.
    """

    async def __call__(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> None: ...
