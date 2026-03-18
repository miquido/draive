from collections.abc import Mapping
from typing import Any, Protocol, Self, final, runtime_checkable

from haiway import Meta, MetaValues, Paginated, Pagination, State

from draive.multimodal.content import Multimodal

__all__ = (
    "Template",
    "TemplateDeclaration",
    "TemplateDefining",
    "TemplateInvalid",
    "TemplateListing",
    "TemplateLoading",
    "TemplateMissing",
)


@final
class TemplateMissing(Exception):
    """Raised when a requested template declaration or content cannot be found.

    Parameters
    ----------
    identifier : str
        Identifier of the missing template.
    """

    __slots__ = ("identifier",)

    def __init__(
        self,
        *,
        identifier: str,
    ) -> None:
        super().__init__(f"Missing template - {identifier}")
        self.identifier: str = identifier


@final
class TemplateInvalid(Exception):
    """Raised when a template declaration is inconsistent with its content.

    Parameters
    ----------
    identifier : str
        Identifier of the invalid template.
    description : str
        Human-readable explanation of the validation failure.
    """

    __slots__ = (
        "description",
        "identifier",
    )

    def __init__(
        self,
        *,
        identifier: str,
        description: str,
    ) -> None:
        super().__init__(f"Invalid template ({description}) - {identifier}")
        self.identifier: str = identifier
        self.description: str = description


@final
class Template(State, serializable=True):
    """Immutable template reference describing how to render multimodal content.

    Templates reference concrete content by identifier and optionally carry default
    argument bindings merged with arguments provided during resolution.

    Parameters
    ----------
    identifier : str
        Unique name of the template to load from a repository.
    arguments : Mapping[str, Self | Multimodal]
        Default argument mapping applied on top of call-time arguments.
    meta : Meta
        Supplemental metadata propagated to the underlying repository implementation.
    """

    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        arguments: Mapping[str, Self | Multimodal] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a template reference.

        Parameters
        ----------
        identifier : str
            Unique name of the template.
        arguments : Mapping[str, Self | Multimodal] | None, optional
            Optional default argument mapping applied when resolving the template.
        meta : Meta | MetaValues | None, optional
            Optional metadata forwarded to repository calls.

        Returns
        -------
        Self
            New template reference instance.
        """
        return cls(
            identifier=identifier,
            arguments=arguments if arguments is not None else {},
            meta=Meta.of(meta),
        )

    identifier: str
    arguments: Mapping[str, Self | Multimodal]
    meta: Meta = Meta.empty

    def with_arguments(
        self,
        **arguments: Self | Multimodal,
    ) -> Self:
        """Return a template with additional default arguments.

        Parameters
        ----------
        **arguments : Self | Multimodal
            Additional argument bindings to merge with existing defaults.

        Returns
        -------
        Self
            Template instance containing merged argument defaults.
        """
        if not arguments:
            return self

        return self.__class__(
            identifier=self.identifier,
            arguments={**self.arguments, **arguments},
            meta=self.meta,
        )

    def with_meta(
        self,
        meta: Meta | MetaValues,
        /,
    ) -> Self:
        """Return a template with merged metadata.

        Parameters
        ----------
        meta : Meta | MetaValues
            Extra metadata values to merge with the existing metadata.

        Returns
        -------
        Self
            Template instance carrying the merged metadata.
        """
        if not meta:
            return self

        return self.__class__(
            identifier=self.identifier,
            arguments=self.arguments,
            meta=self.meta.merged_with(meta),
        )


@final
class TemplateDeclaration(State, serializable=True):
    """Immutable template declaration describing user-facing template metadata.

    Declarations surface template descriptions and declared variables without
    requiring the template content itself to be loaded.

    Parameters
    ----------
    identifier : str
        Unique name of the template declaration.
    description : str | None
        Optional human-readable summary of the template purpose.
    variables : Mapping[str, str]
        Mapping of variable names to description strings.
    meta : Meta
        Supplemental metadata preserved when invoking repository operations.
    """

    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        description: str | None = None,
        variables: Mapping[str, str] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a template declaration.

        Parameters
        ----------
        identifier : str
            Unique name of the template declaration.
        description : str | None, optional
            Optional human-readable description.
        variables : Mapping[str, str] | None, optional
            Mapping of variable names to their descriptions.
        meta : Meta | MetaValues | None, optional
            Optional metadata forwarded to repository calls.

        Returns
        -------
        Self
            New template declaration instance.
        """
        return cls(
            identifier=identifier,
            description=description,
            variables=variables if variables is not None else {},
            meta=Meta.of(meta),
        )

    identifier: str
    description: str | None = None
    variables: Mapping[str, str]
    meta: Meta = Meta.empty


@runtime_checkable
class TemplateListing(Protocol):
    """Protocol for listing available template declarations.

    Returns paginated template declarations from a repository or registry.
    Additional keyword arguments allow integrations to expose backend-specific
    controls without weakening the common callable shape.
    """

    async def __call__(
        self,
        pagination: Pagination | None,
        **extra: Any,
    ) -> Paginated[TemplateDeclaration]:
        """List template declarations.

        Parameters
        ----------
        pagination : Pagination | None
            Pagination cursor and page-size information for the listing request.
            When ``None``, the implementation should use its default pagination
            behavior.
        **extra : Any
            Backend-specific options accepted by the listing implementation.

        Returns
        -------
        Paginated[TemplateDeclaration]
            Paginated collection of template declarations.
        """
        ...


@runtime_checkable
class TemplateLoading(Protocol):
    """Protocol for loading template content for a given identifier.

    Implementations load raw template content and may use metadata and
    additional keyword arguments to drive repository-specific lookup behavior.
    """

    async def __call__(
        self,
        identifier: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None:
        """Load template content for a given identifier.

        Parameters
        ----------
        identifier : str
            Unique template identifier to load.
        meta : Meta
            Metadata forwarded with the loading request.
        **extra : Any
            Backend-specific options accepted by the loading implementation.

        Returns
        -------
        str | None
            Template content when found, otherwise ``None``.
        """
        ...


@runtime_checkable
class TemplateDefining(Protocol):
    """Protocol for upserting template content and metadata.

    Implementations persist template content together with declaration metadata,
    creating or replacing the stored definition for the given identifier.
    """

    async def __call__(
        self,
        identifier: str,
        description: str | None,
        content: str,
        variables: Mapping[str, str],
        meta: Meta,
        **extra: Any,
    ) -> None:
        """Create or update a template definition.

        Parameters
        ----------
        identifier : str
            Unique template identifier to define.
        description : str | None
            Optional human-readable description of the template.
        content : str
            Raw template source to persist.
        variables : Mapping[str, str]
            Mapping of declared variable names to their descriptions.
        meta : Meta
            Metadata forwarded with the defining request.
        **extra : Any
            Backend-specific options accepted by the defining implementation.
        """
        ...
