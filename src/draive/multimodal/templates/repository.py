import json
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Self, final, overload

from haiway import (
    META_EMPTY,
    File,
    FileAccess,
    Immutable,
    Meta,
    State,
    ctx,
    statemethod,
)

from draive.multimodal.content import Multimodal, MultimodalContent
from draive.multimodal.templates.types import (
    Template,
    TemplateDeclaration,
    TemplateDefining,
    TemplateListing,
    TemplateLoading,
    TemplateMissing,
)
from draive.multimodal.templates.variables import (
    parse_template_variables,
    resolve_multimodal_template,
    resolve_text_template,
)

__all__ = ("TemplatesRepository",)


async def _empty_listing(
    **extra: Any,
) -> Sequence[TemplateDeclaration]:
    return ()


async def _none_loading(
    identifier: str,
    meta: Meta,
    **extra: Any,
) -> str | None:
    return None


async def _noop_defining(
    identifier: str,
    description: str | None,
    content: str,
    variables: Mapping[str, str],
    meta: Meta,
    **extra: Any,
) -> None:
    pass


@final
class TemplatesRepository(State):
    """State facade orchestrating template storage and resolution backends.

    A repository aggregates callables responsible for listing template metadata,
    loading template bodies, and persisting updates. Convenience constructors
    supply in-memory and file-backed implementations suitable for tests and
    lightweight deployments.
    """

    @classmethod
    def volatile(
        cls,
        **templates: str,
    ) -> Self:
        """Create an in-memory repository seeded with template contents.

        Parameters
        ----------
        **templates
            Mapping of template identifier to template content.

        Returns
        -------
        Self
            Repository instance with volatile storage.
        """
        volatile_storage: VolatileStorage = VolatileStorage(
            _declarations={
                identifier: TemplateDeclaration(
                    identifier=identifier,
                    variables={
                        variable: variable for variable in parse_template_variables(content)
                    },
                    description=None,
                    meta=META_EMPTY,
                )
                for identifier, content in templates.items()
            },
            _contents=templates,
        )

        return cls(
            listing=volatile_storage.listing,
            loading=volatile_storage.loading,
            defining=volatile_storage.defining,
            meta=Meta({"source": "volatile"}),
        )

    @classmethod
    def file(
        cls,
        path: Path | str,
    ) -> Self:
        """Create a repository using a JSON file for persistence.

        Parameters
        ----------
        path
            Filesystem path where templates will be stored and loaded.

        Returns
        -------
        Self
            Repository instance backed by the given file.
        """
        file_storage: FileStorage = FileStorage(path=path)

        return cls(
            listing=file_storage.listing,
            loading=file_storage.loading,
            defining=file_storage.defining,
            meta=Meta({"source": str(path)}),
        )

    @overload
    @classmethod
    async def templates(
        cls,
        **extra: Any,
    ) -> Sequence[TemplateDeclaration]: ...

    @overload
    async def templates(
        self,
        **extra: Any,
    ) -> Sequence[TemplateDeclaration]: ...

    @statemethod
    async def templates(
        self,
        **extra: Any,
    ) -> Sequence[TemplateDeclaration]:
        """List available template declarations.

        Parameters
        ----------
        **extra
            Extra arguments forwarded to the underlying listing callable.

        Returns
        -------
        Sequence[TemplateDeclaration]
            Collection of available template declarations.
        """
        return await self.listing(**extra)

    @overload
    @classmethod
    async def resolve(
        cls,
        content: Template | Multimodal,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Multimodal] | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...

    @overload
    async def resolve(
        self,
        content: Template | Multimodal,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Multimodal] | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...

    @statemethod
    async def resolve(
        self,
        content: Template | Multimodal,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Multimodal] | None = None,
        **extra: Any,
    ) -> MultimodalContent:
        """Resolve a template into multimodal content.

        Parameters
        ----------
        content
            Template reference to resolve or straight result content.
        default
            Fallback template body used when the repository yields no content.
        arguments
            Additional arguments merged with the template defaults before rendering.
        **extra
            Extra arguments forwarded to the underlying loading callable.

        Returns
        -------
        MultimodalContent
            Rendered multimodal content.

        Raises
        ------
        TemplateMissing
            If neither repository content nor ``default`` is available.
        """
        if not isinstance(content, Template):
            return MultimodalContent.of(content)

        loaded: str | None = await self.loading(
            content.identifier,
            meta=content.meta,
            **extra,
        )

        merged_arguments: Mapping[str, Multimodal]
        if arguments:
            merged_arguments = {**arguments, **content.arguments}

        else:
            merged_arguments = content.arguments

        if loaded is not None:
            return resolve_multimodal_template(
                loaded,
                arguments=merged_arguments,
            )

        elif default is not None:
            return resolve_multimodal_template(
                default,
                arguments=merged_arguments,
            )

        else:
            raise TemplateMissing(identifier=content.identifier)

    @overload
    @classmethod
    async def resolve_str(
        cls,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Multimodal] | None = None,
        **extra: Any,
    ) -> str: ...

    @overload
    async def resolve_str(
        self,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Multimodal] | None = None,
        **extra: Any,
    ) -> str: ...

    @statemethod
    async def resolve_str(
        self,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Multimodal] | None = None,
        **extra: Any,
    ) -> str:
        """Resolve a template into a text string.

        Parameters
        ----------
        content
            Template reference or raw string to return as-is.
        default
            Fallback template body used when the repository yields no content.
        arguments
            Additional arguments merged with the template defaults before rendering.
        **extra
            Extra arguments forwarded to the underlying loading callable.

        Returns
        -------
        str
            Rendered text content.

        Raises
        ------
        TemplateMissing
            If neither repository content nor ``default`` is available.
        """
        if not isinstance(content, Template):
            if arguments:
                return content.format_map({key: str(value) for key, value in arguments.items()})

            else:
                return content

        loaded: str | None = await self.loading(
            content.identifier,
            meta=content.meta,
            **extra,
        )

        merged_arguments: Mapping[str, Multimodal]
        if arguments:
            merged_arguments = {**arguments, **content.arguments}

        else:
            merged_arguments = content.arguments

        if loaded is not None:
            return resolve_text_template(
                loaded,
                arguments=merged_arguments,
            )

        elif default is not None:
            return resolve_text_template(
                default,
                arguments=merged_arguments,
            )

        else:
            raise TemplateMissing(identifier=content.identifier)

    @statemethod
    async def load(
        self,
        template: Template,
        /,
        **extra: Any,
    ) -> str:
        """Load raw template content without applying argument substitution.

        Parameters
        ----------
        template
            Template reference to load.
        **extra
            Extra arguments forwarded to the underlying loading callable.

        Returns
        -------
        str
            Raw template body retrieved from the repository.

        Raises
        ------
        TemplateMissing
            If the template content is not available.
        """
        loaded: str | None = await self.loading(
            template.identifier,
            meta=template.meta,
            **extra,
        )

        if loaded is None:
            raise TemplateMissing(identifier=template.identifier)

        return loaded

    @overload
    @classmethod
    async def define(
        cls,
        template: TemplateDeclaration,
        /,
        *,
        content: str,
        **extra: Any,
    ) -> None: ...

    @overload
    async def define(
        self,
        template: TemplateDeclaration,
        /,
        *,
        content: str,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def define(
        self,
        template: TemplateDeclaration,
        /,
        *,
        content: str,
        **extra: Any,
    ) -> None:
        """Persist or update a template declaration and its content.

        Parameters
        ----------
        template
            Declaration describing the template metadata to be stored.
        content
            Template body to associate with the declaration.
        **extra
            Extra arguments forwarded to the underlying defining callable.
        """
        await self.defining(
            identifier=template.identifier,
            description=template.description,
            variables=template.variables,
            content=content,
            meta=template.meta,
            **extra,
        )

    listing: TemplateListing = _empty_listing
    loading: TemplateLoading = _none_loading
    defining: TemplateDefining = _noop_defining
    meta: Meta = META_EMPTY


class VolatileStorage(Immutable):
    _declarations: MutableMapping[str, TemplateDeclaration]
    _contents: MutableMapping[str, str]

    async def listing(
        self,
        **extra: Any,
    ) -> Sequence[TemplateDeclaration]:
        return tuple(self._declarations.values())

    async def loading(
        self,
        identifier: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None:
        return self._contents.get(identifier)

    async def defining(
        self,
        identifier: str,
        description: str | None,
        content: str,
        variables: Mapping[str, str],
        meta: Meta,
        **extra: Any,
    ) -> None:
        self._declarations[identifier] = TemplateDeclaration(
            identifier=identifier,
            description=description,
            variables=variables,
            meta=meta,
        )
        self._contents[identifier] = content


class FileStorage(Immutable):
    _path: Path
    _declarations: MutableMapping[str, TemplateDeclaration] | None
    _contents: MutableMapping[str, str] | None

    def __init__(
        self,
        path: Path | str,
    ) -> None:
        if isinstance(path, str):
            object.__setattr__(
                self,
                "_path",
                Path(path),
            )

        else:
            object.__setattr__(
                self,
                "_path",
                path,
            )

        object.__setattr__(
            self,
            "_declarations",
            None,
        )
        object.__setattr__(self, "_contents", None)

    async def listing(
        self,
        **extra: Any,
    ) -> Sequence[TemplateDeclaration]:
        if self._declarations is None:
            await self._load_file()

        assert self._declarations is not None  # nosec: B101
        return tuple(self._declarations.values())

    async def loading(
        self,
        identifier: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None:
        if self._contents is None:
            await self._load_file()

        assert self._contents is not None  # nosec: B101
        return self._contents.get(identifier)

    async def defining(
        self,
        identifier: str,
        description: str | None,
        content: str,
        variables: Mapping[str, str],
        meta: Meta,
        **extra: Any,
    ) -> None:
        if self._contents is None or self._declarations is None:
            await self._load_file()

        self._declarations[identifier] = TemplateDeclaration(  # pyright: ignore[reportOptionalSubscript]
            identifier=identifier,
            description=description,
            variables=variables,
            meta=meta,
        )
        self._contents[identifier] = content  # pyright: ignore[reportOptionalSubscript]
        await self._save_file()

    async def _load_file(self) -> None:
        file_contents: bytes
        async with ctx.disposables(FileAccess.open(self._path, create=True)):
            file_contents = await File.read()

        if not file_contents.strip():
            file_contents = b"[]"

        declarations: MutableMapping[str, TemplateDeclaration] = {}
        contents: MutableMapping[str, str] = {}
        try:
            match json.loads(file_contents):
                case [*elements]:
                    for element in elements:
                        match element:
                            case {
                                "identifier": str() as identifier,
                                "description": str() | None as description,
                                "variables": {**variables},
                                "content": str() as content,
                                "meta": {**meta},
                            }:
                                declarations[identifier] = TemplateDeclaration(
                                    identifier=identifier,
                                    variables=variables,
                                    description=description,
                                    meta=Meta.of(meta),
                                )
                                contents[identifier] = content

                            case _:  # skip with warning
                                ctx.log_warning(
                                    "Invalid templates file storage element, skipping..."
                                )

                case _:  # empty storage with error
                    ctx.log_error("Invalid templates file storage, using empty...")

        except Exception as exc:
            ctx.log_error(
                "Invalid templates file storage, using empty...",
                exception=exc,
            )

        object.__setattr__(
            self,
            "_declarations",
            declarations,
        )

        object.__setattr__(
            self,
            "_contents",
            contents,
        )

    async def _save_file(self) -> None:
        if self._declarations is None or self._contents is None:
            return  # nothing to save

        file_contents: bytes = json.dumps(
            [
                {
                    "identifier": declaration.identifier,
                    "description": declaration.description,
                    "variables": declaration.variables,
                    "content": self._contents[declaration.identifier],
                    "meta": declaration.meta,
                }
                for declaration in self._declarations.values()
            ]
        ).encode()

        async with ctx.disposables(FileAccess.open(self._path, create=True)):
            await File.write(file_contents)
