from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, final, runtime_checkable

from haiway import META_EMPTY, Meta, MetaValues, State

from draive.multimodal.content import Multimodal

__all__ = (
    "Template",
    "TemplateDeclaration",
    "TemplateDefining",
    "TemplateListing",
    "TemplateLoading",
    "TemplateMissing",
)


class TemplateMissing(Exception):
    """Raised when a requested template declaration or content cannot be found.

    Parameters
    ----------
    identifier
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
class Template(State):
    """Immutable template reference describing how to render multimodal content.

    Templates reference concrete content by identifier and optionally carry default
    argument bindings that will be merged with arguments provided during resolution.

    Parameters
    ----------
    identifier
        Unique name of the template to load from a repository.
    arguments
        Default argument mapping applied on top of call-time arguments.
    meta
        Supplemental metadata propagated to the underlying repository implementation.
    """

    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        arguments: Mapping[str, Multimodal] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a template reference with optional defaults.

        Parameters
        ----------
        identifier
            Unique name of the template.
        arguments
            Default argument mapping applied when resolving the template.
        meta
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
    arguments: Mapping[str, Multimodal]
    meta: Meta

    def with_arguments(
        self,
        **arguments: Multimodal,
    ) -> Self:
        """Return a new template augmented with additional default arguments.

        Parameters
        ----------
        **arguments
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
        """Return a new template with metadata merged into the current meta.

        Parameters
        ----------
        meta
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
class TemplateDeclaration(State):
    """Immutable template declaration describing user-facing template metadata.

    Declarations surface template descriptions and declared variables without
    requiring the template content itself to be loaded.

    Parameters
    ----------
    identifier
        Unique name of the template declaration.
    description
        Optional human-readable summary of the template purpose.
    variables
        Mapping of variable names to description strings.
    meta
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
        """Create a template declaration with optional description and variables.

        Parameters
        ----------
        identifier
            Unique name of the template declaration.
        description
            Optional human-readable description.
        variables
            Mapping of variable names to their descriptions.
        meta
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
    meta: Meta = META_EMPTY


@runtime_checkable
class TemplateListing(Protocol):
    """Callable protocol returning available template declarations."""

    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[TemplateDeclaration]: ...


@runtime_checkable
class TemplateLoading(Protocol):
    """Callable protocol loading template content for the given identifier."""

    async def __call__(
        self,
        identifier: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None: ...


@runtime_checkable
class TemplateDefining(Protocol):
    """Callable protocol that upserts template content and metadata."""

    async def __call__(
        self,
        identifier: str,
        description: str | None,
        content: str,
        variables: Mapping[str, str],
        meta: Meta,
        **extra: Any,
    ) -> None: ...
