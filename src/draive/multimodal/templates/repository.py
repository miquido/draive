import json
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Self, cast, final, overload

from haiway import (
    File,
    FileAccess,
    Immutable,
    Meta,
    Paginated,
    Pagination,
    State,
    ctx,
    statemethod,
)

from draive.multimodal.content import Multimodal, MultimodalContent
from draive.multimodal.templates.types import (
    Template,
    TemplateDeclaration,
    TemplateDefining,
    TemplateInvalid,
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
    pagination: Pagination | None,
    **extra: Any,
) -> Paginated[TemplateDeclaration]:
    _ = extra
    return Paginated[TemplateDeclaration].of(
        (),
        pagination=Pagination.of(limit=0),
    )


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
    """Template storage and resolution backend.

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
        **templates : str
            Template contents keyed by template identifier.

        Returns
        -------
        repository : Self
            Repository instance backed by process-local mutable storage.
        """
        volatile_storage: VolatileStorage = VolatileStorage(
            _declarations={
                identifier: TemplateDeclaration(
                    identifier=identifier,
                    variables={
                        variable: variable for variable in parse_template_variables(content)
                    },
                    description=None,
                    meta=Meta.empty,
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
        path : Path | str
            Filesystem path used to load and persist template declarations and
            their contents.

        Returns
        -------
        repository : Self
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
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[TemplateDeclaration]: ...

    @overload
    async def templates(
        self,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[TemplateDeclaration]: ...

    @statemethod
    async def templates(
        self,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[TemplateDeclaration]:
        """List available template declarations.

        Parameters
        ----------
        pagination : Pagination | None, optional
            Pagination settings controlling the declaration page to return. When
            omitted, the repository default is used.
        **extra : Any
            Extra arguments forwarded to the underlying listing callable.

        Returns
        -------
        declarations : Paginated[TemplateDeclaration]
            Collection of available template declarations.
        """
        return await self._listing(
            pagination=pagination,
            **extra,
        )

    @overload
    @classmethod
    async def resolve(
        cls,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Template | Multimodal] | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...

    @overload
    async def resolve(
        self,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Template | Multimodal] | None = None,
        **extra: Any,
    ) -> MultimodalContent: ...

    @statemethod
    async def resolve(
        self,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Template | Multimodal] | None = None,
        **extra: Any,
    ) -> MultimodalContent:
        """Resolve a template into multimodal content.

        Parameters
        ----------
        content : Template | str
            Template reference to resolve or raw template source to render
            directly.
        default : str | None, optional
            Fallback template body used when the repository yields no content.
        arguments : Mapping[str, Template | Multimodal] | None, optional
            Additional template arguments merged with the template declaration
            arguments before rendering. Nested `Template` values are resolved
            recursively.
        **extra : Any
            Extra arguments forwarded to the underlying loading callable.

        Returns
        -------
        resolved : MultimodalContent
            Rendered multimodal content.

        Raises
        ------
        TemplateMissing
            If neither repository content nor ``default`` is available.
        """
        if isinstance(content, str):
            return resolve_multimodal_template(
                content,
                arguments=await self._resolve_arguments(
                    arguments,
                    **extra,
                ),
            )

        loaded: str | None = await self._loading(
            content.identifier,
            meta=content.meta,
            **extra,
        )

        if loaded is not None:
            return resolve_multimodal_template(
                loaded,
                arguments=await self._resolve_arguments(
                    {**content.arguments, **arguments} if arguments else content.arguments,
                    **extra,
                ),
            )

        elif default is not None:
            return resolve_multimodal_template(
                default,
                arguments=await self._resolve_arguments(
                    {**content.arguments, **arguments} if arguments else content.arguments,
                    **extra,
                ),
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
        arguments: Mapping[str, Template | Multimodal] | None = None,
        **extra: Any,
    ) -> str: ...

    @overload
    async def resolve_str(
        self,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Template | Multimodal] | None = None,
        **extra: Any,
    ) -> str: ...

    @statemethod
    async def resolve_str(
        self,
        content: Template | str,
        /,
        *,
        default: str | None = None,
        arguments: Mapping[str, Template | Multimodal] | None = None,
        **extra: Any,
    ) -> str:
        """Resolve a template into a text string.

        Parameters
        ----------
        content : Template | str
            Template reference to resolve or raw template source to render
            directly.
        default : str | None, optional
            Fallback template body used when the repository yields no content.
        arguments : Mapping[str, Template | Multimodal] | None, optional
            Additional template arguments merged with the template declaration
            arguments before rendering. Nested `Template` values are resolved
            recursively.
        **extra : Any
            Extra arguments forwarded to the underlying loading callable.

        Returns
        -------
        resolved : str
            Rendered text content.

        Raises
        ------
        TemplateMissing
            If neither repository content nor ``default`` is available.
        """
        if isinstance(content, str):
            return resolve_text_template(
                content,
                arguments=await self._resolve_arguments(
                    arguments,
                    **extra,
                ),
            )

        loaded: str | None = await self._loading(
            content.identifier,
            meta=content.meta,
            **extra,
        )

        if loaded is not None:
            return resolve_text_template(
                loaded,
                arguments=await self._resolve_arguments(
                    {**content.arguments, **arguments} if arguments else content.arguments,
                    **extra,
                ),
            )

        elif default is not None:
            return resolve_text_template(
                default,
                arguments=await self._resolve_arguments(
                    {**content.arguments, **arguments} if arguments else content.arguments,
                    **extra,
                ),
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
        template : Template
            Template reference to load.
        **extra : Any
            Extra arguments forwarded to the underlying loading callable.

        Returns
        -------
        content : str
            Raw template body retrieved from the repository.

        Raises
        ------
        TemplateMissing
            If the template content is not available.
        """
        loaded: str | None = await self._loading(
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
        template : TemplateDeclaration
            Declaration describing the template metadata to be stored.
        content : str
            Template body to associate with the declaration.
        **extra : Any
            Extra arguments forwarded to the underlying defining callable.

        Returns
        -------
        None
            Returns after the template declaration and content are persisted.

        Raises
        ------
        TemplateInvalid
            If the declared template variables do not match the variables
            parsed from `content`.
        """
        parsed_variables: tuple[str, ...] = tuple(dict.fromkeys(parse_template_variables(content)))
        declared_variables: tuple[str, ...] = tuple(template.variables.keys())
        if set(parsed_variables) != set(declared_variables):
            raise TemplateInvalid(
                identifier=template.identifier,
                description=(
                    "declared variables do not match template content "
                    f"(declared={declared_variables}, parsed={parsed_variables})"
                ),
            )

        await self._defining(
            identifier=template.identifier,
            description=template.description,
            variables=template.variables,
            content=content,
            meta=template.meta,
            **extra,
        )

    _listing: TemplateListing
    _loading: TemplateLoading
    _defining: TemplateDefining
    meta: Meta

    def __init__(
        self,
        listing: TemplateListing = _empty_listing,
        loading: TemplateLoading = _none_loading,
        defining: TemplateDefining = _noop_defining,
        meta: Meta = Meta.empty,
    ) -> None:
        super().__init__(
            _listing=listing,
            _loading=loading,
            _defining=defining,
            meta=meta,
        )

    async def _resolve_arguments(
        self,
        arguments: Mapping[str, Template | Multimodal] | None,
        **extra: Any,
    ) -> Mapping[str, Multimodal]:
        if arguments is None:
            return {}

        arguments = dict(arguments)
        for key, value in arguments.items():
            if isinstance(value, Exception):
                raise value

            elif isinstance(value, Template):
                arguments[key] = RecursionError("Recursive template resolution is not supported")  # pyright: ignore[reportArgumentType]
                resolved: MultimodalContent = await self.resolve(
                    value,
                    arguments=arguments,
                    **extra,
                )
                arguments[key] = resolved

        return cast(Mapping[str, Multimodal], arguments)


def _paginate_declarations(
    declarations: Sequence[TemplateDeclaration],
    *,
    pagination: Pagination | None,
    source: str,
) -> Paginated[TemplateDeclaration]:
    pagination = pagination or Pagination.of(limit=32)

    if pagination.limit <= 0:
        return Paginated[TemplateDeclaration].of(
            (),
            pagination=pagination.with_token(None),
        )

    start: int
    if pagination.token is None:
        start = 0

    elif isinstance(pagination.token, str):
        if not pagination.token.startswith("templates:"):
            raise ValueError(f"Invalid {source} templates pagination token")

        try:
            start = max(int(pagination.token.split(":", 1)[1]), 0)

        except ValueError as exc:
            raise ValueError(f"Invalid {source} templates pagination token") from exc

    elif isinstance(pagination.token, int):
        start = max(pagination.token, 0)

    else:
        raise ValueError(f"Invalid {source} templates pagination token")

    end: int = start + pagination.limit
    next_token: str | None = None
    if end < len(declarations):
        next_token = f"templates:{end}"

    return Paginated[TemplateDeclaration].of(
        declarations[start:end],
        pagination=pagination.with_token(next_token),
    )


class VolatileStorage(Immutable):
    _declarations: MutableMapping[str, TemplateDeclaration]
    _contents: MutableMapping[str, str]

    async def listing(
        self,
        pagination: Pagination | None,
        **extra: Any,
    ) -> Paginated[TemplateDeclaration]:
        _ = extra
        return _paginate_declarations(
            tuple(self._declarations.values()),
            pagination=pagination,
            source="volatile",
        )

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
        pagination: Pagination | None,
        **extra: Any,
    ) -> Paginated[TemplateDeclaration]:
        _ = extra
        if self._declarations is None:
            await self._load_file()

        assert self._declarations is not None  # nosec: B101
        return _paginate_declarations(
            tuple(self._declarations.values()),
            pagination=pagination,
            source="file",
        )

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
                        try:
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

                        except Exception as exc:
                            ctx.log_warning(
                                "Invalid templates file storage element, skipping...",
                                exception=exc,
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
