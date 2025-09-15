import json
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Self, final, overload

from haiway import META_EMPTY, File, FileAccess, Immutable, Meta, State, ctx, statemethod

from draive.models.instructions.types import (
    Instructions,
    InstructionsArgumentDeclaration,
    InstructionsDeclaration,
    InstructionsDefining,
    InstructionsListing,
    InstructionsLoading,
    InstructionsMissing,
    InstructionsRemoving,
)
from draive.models.types import ModelInstructions

__all__ = (
    "InstructionsRepository",
    "ResolveableInstructions",
)


ResolveableInstructions = Instructions | ModelInstructions


async def _empty_listing(
    **extra: Any,
) -> Sequence[InstructionsDeclaration]:
    return ()


async def _none_loading(
    name: str,
    meta: Meta,
    **extra: Any,
) -> str | None:
    return None


async def _noop_defining(
    declaration: InstructionsDeclaration,
    content: str,
    **extra: Any,
) -> None:
    pass


async def _noop_removing(
    name: str,
    meta: Meta,
    **extra: Any,
) -> None:
    pass


@final
class InstructionsRepository(State):
    """Repository facade for listing, resolving, defining, and removing instructions.

    Backed by pluggable callables to support different storage backends (in-memory,
    file-based, remote). Provides convenient statemethods usable via class or instance.
    """

    @classmethod
    def volatile(
        cls,
        **instructions: str,
    ) -> Self:
        """Create a repository backed by in-memory volatile storage.

        Parameters
        ----------
        instructions : dict[str, str]
            Mapping of instruction name to its content.

        Returns
        -------
        InstructionsRepository
            Repository facade configured with volatile in-memory storage.
        """
        volatile_storage: InstructionsVolatileStorage = InstructionsVolatileStorage(
            _declarations={
                key: InstructionsDeclaration(
                    name=key,
                    arguments=(),
                    description=None,
                    meta=META_EMPTY,
                )
                for key in instructions.keys()
            },
            _contents=instructions,
        )

        return cls(
            listing=volatile_storage.listing,
            loading=volatile_storage.loading,
            defining=volatile_storage.defining,
            removing=volatile_storage.removing,
            meta=Meta({"source": "volatile"}),
        )

    @classmethod
    def file(
        cls,
        path: Path | str,
    ) -> Self:
        """Create a repository backed by a JSON file storage.

        Parameters
        ----------
        path : Path | str
            Path to the storage file. The file is created on first write.

        Returns
        -------
        InstructionsRepository
            Repository facade configured with file-backed storage.
        """
        file_storage: InstructionsFileStorage = InstructionsFileStorage(path=path)

        return cls(
            listing=file_storage.listing,
            loading=file_storage.loading,
            defining=file_storage.defining,
            removing=file_storage.removing,
            meta=Meta({"source": str(path)}),
        )

    @overload
    @classmethod
    async def available_instructions(
        cls,
        **extra: Any,
    ) -> Sequence[InstructionsDeclaration]: ...

    @overload
    async def available_instructions(
        self,
        **extra: Any,
    ) -> Sequence[InstructionsDeclaration]: ...

    @statemethod
    async def available_instructions(
        self,
        **extra: Any,
    ) -> Sequence[InstructionsDeclaration]:
        """List available instruction declarations from the backend.

        Parameters
        ----------
        **extra : Any
            Extra keyword arguments forwarded to the underlying listing callable.

        Returns
        -------
        Sequence[InstructionsDeclaration]
            Declarations available in the configured storage backend.
        """
        return await self.listing(
            **extra,
        )

    @overload
    @classmethod
    async def resolve(
        cls,
        instructions: ResolveableInstructions | None,
        /,
        *,
        arguments: Mapping[str, str | int | float] | None = None,
        default: str | None = None,
        **extra: Any,
    ) -> ModelInstructions: ...

    @overload
    async def resolve(
        self,
        instructions: ResolveableInstructions | None,
        /,
        *,
        arguments: Mapping[str, str | int | float] | None = None,
        default: str | None = None,
        **extra: Any,
    ) -> ModelInstructions: ...

    @statemethod
    async def resolve(
        self,
        instructions: ResolveableInstructions | None,
        /,
        *,
        arguments: Mapping[str, str | int | float] | None = None,
        default: str | None = None,
        **extra: Any,
    ) -> ModelInstructions:
        """Resolve instructions to a formatted string.

        - If ``instructions`` is ``None``: returns ``default`` formatted (or empty string).
        - If a string is provided: formats it with ``arguments``.
        - If an ``Instructions`` reference is provided: loads content and formats with
          merged arguments. Falls back to ``default`` if content is missing; otherwise
          raises ``InstructionsMissing``.

        Parameters
        ----------
        instructions : ResolveableInstructions | None
            Direct content, a reference, or ``None``.
        arguments : Mapping[str, str | int | float] | None, optional
            Template variables.
        default : str | None, optional
            Fallback content when resolution fails or input is ``None``.
        **extra : Any
            Extra kwargs forwarded to backends.

        Returns
        -------
        str
            Resolved instructions content.

        Raises
        ------
        InstructionsMissing
            Raised when the referenced instructions cannot be loaded and no default
            fallback is provided.
        """
        if instructions is None:
            return (
                default.format_map(arguments if arguments is not None else {})
                if default is not None
                else ""
            )

        if isinstance(instructions, str):
            return instructions.format_map(arguments if arguments is not None else {})

        loaded_instructions: str | None = await self.loading(
            instructions.name,
            meta=instructions.meta,
            **extra,
        )

        if loaded_instructions is not None:
            return loaded_instructions.format_map(
                {
                    **instructions.arguments,
                    **(arguments if arguments is not None else {}),
                },
            )

        elif default is not None:
            return default.format_map(
                {
                    **instructions.arguments,
                    **(arguments if arguments is not None else {}),
                },
            )

        else:
            raise InstructionsMissing(name=instructions.name)

    @statemethod
    async def load(
        self,
        instructions: Instructions,
        /,
        **extra: Any,
    ) -> ModelInstructions:
        """Load the raw instructions content from the backend.

        Parameters
        ----------
        instructions : Instructions
            Instructions reference identifying what to load.
        **extra : Any
            Extra keyword arguments forwarded to the loading callable.

        Returns
        -------
        str
            Loaded instructions content.

        Raises
        ------
        InstructionsMissing
            Raised when the repository backend has no content for the reference.
        """
        loaded_instructions: str | None = await self.loading(
            instructions.name,
            meta=instructions.meta,
            **extra,
        )
        if loaded_instructions is None:
            raise InstructionsMissing(name=instructions.name)

        return loaded_instructions

    @overload
    @classmethod
    async def define(
        cls,
        instructions: InstructionsDeclaration | str,
        /,
        *,
        content: str,
        **extra: Any,
    ) -> None: ...

    @overload
    async def define(
        self,
        instructions: InstructionsDeclaration | str,
        /,
        *,
        content: str,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def define(
        self,
        instructions: InstructionsDeclaration | str,
        /,
        *,
        content: str,
        **extra: Any,
    ) -> None:
        """Create or update an instructions template in the backend.

        Parameters
        ----------
        instructions : InstructionsDeclaration | str
            Template declaration or name describing what to define.
        content : str
            Template body that should be stored.
        **extra : Any
            Extra keyword arguments forwarded to the defining callable.

        Returns
        -------
        None
            This method performs a side effect on the configured backend.
        """
        declaration: InstructionsDeclaration
        if isinstance(instructions, str):
            declaration = InstructionsDeclaration.of(instructions)

        else:
            declaration = instructions

        await self.defining(
            declaration=declaration,
            content=content,
            **extra,
        )

    @overload
    @classmethod
    async def remove(
        cls,
        instructions: InstructionsDeclaration,
        /,
        **extra: Any,
    ) -> None: ...

    @overload
    async def remove(
        self,
        instructions: InstructionsDeclaration,
        /,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def remove(
        self,
        instructions: InstructionsDeclaration,
        /,
        **extra: Any,
    ) -> None:
        """Remove an instructions template from the backend.

        Parameters
        ----------
        instructions : InstructionsDeclaration
            Template declaration identifying what to remove.
        **extra : Any
            Extra keyword arguments forwarded to the removing callable.

        Returns
        -------
        None
            This method performs a side effect on the configured backend.
        """
        await self.removing(
            name=instructions.name,
            meta=instructions.meta,
            **extra,
        )

    listing: InstructionsListing = _empty_listing
    loading: InstructionsLoading = _none_loading
    defining: InstructionsDefining = _noop_defining
    removing: InstructionsRemoving = _noop_removing
    meta: Meta = META_EMPTY


class InstructionsVolatileStorage(Immutable):
    """Internal in-memory storage used by ``InstructionsRepository.volatile``."""

    _declarations: MutableMapping[str, InstructionsDeclaration]
    _contents: MutableMapping[str, str]

    async def listing(
        self,
        **extra: Any,
    ) -> Sequence[InstructionsDeclaration]:
        return tuple(self._declarations.values())

    async def loading(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None:
        return self._contents.get(name)

    async def defining(
        self,
        declaration: InstructionsDeclaration,
        content: str,
        **extra: Any,
    ) -> None:
        self._declarations[declaration.name] = declaration
        self._contents[declaration.name] = content

    async def removing(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> None:
        del self._declarations[name]
        del self._contents[name]


class InstructionsFileStorage(Immutable):
    """Internal file-backed storage used by ``InstructionsRepository.file``.

    Stores a JSON array of objects with fields: name, arguments, content, description, meta.
    """

    _path: Path
    _declarations: MutableMapping[str, InstructionsDeclaration] | None
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
    ) -> Sequence[InstructionsDeclaration]:
        if self._declarations is None:
            await self._load_file()

        assert self._declarations is not None  # nosec: B101
        return tuple(self._declarations.values())

    async def loading(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None:
        if self._contents is None:
            await self._load_file()

        assert self._contents is not None  # nosec: B101
        return self._contents.get(name)

    async def defining(
        self,
        declaration: InstructionsDeclaration,
        content: str,
        **extra: Any,
    ) -> None:
        if self._contents is None or self._declarations is None:
            await self._load_file()

        self._declarations[declaration.name] = declaration  # pyright: ignore[reportOptionalSubscript]
        self._contents[declaration.name] = content  # pyright: ignore[reportOptionalSubscript]
        await self._save_file()

    async def removing(
        self,
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> None:
        if self._contents is None or self._declarations is None:
            await self._load_file()

        del self._declarations[name]  # pyright: ignore[reportOptionalSubscript]
        del self._contents[name]  # pyright: ignore[reportOptionalSubscript]
        await self._save_file()

    async def _load_file(self) -> None:
        file_contents: bytes
        async with ctx.disposables(FileAccess.open(self._path)):
            file_contents = await File.read()

        declarations: MutableMapping[str, InstructionsDeclaration] = {}
        contents: MutableMapping[str, str] = {}
        try:
            match json.loads(file_contents):
                case [*elements]:
                    for element in elements:
                        match element:
                            case {
                                "name": str() as name,
                                "arguments": [*arguments],
                                "content": str() as content,
                                "description": str() | None as description,
                                "meta": {**meta},
                            }:
                                declarations[name] = InstructionsDeclaration(
                                    name=name,
                                    arguments=[
                                        InstructionsArgumentDeclaration.from_mapping(argument)
                                        for argument in arguments
                                    ],
                                    description=description,
                                    meta=Meta.of(meta),
                                )
                                contents[name] = content

                            case _:
                                # skip with warning
                                ctx.log_warning(
                                    "Invalid file instructions storage element, skipping..."
                                )

                case _:
                    # empty storage with error
                    ctx.log_error("Invalid file instructions storage, using empty storage...")

        except Exception as exc:
            ctx.log_error(
                "Invalid file instructions storage, using empty storage...",
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
                    "name": declaration.name,
                    "arguments": [argument.to_mapping() for argument in declaration.arguments],
                    "content": self._contents[declaration.name],
                    "description": declaration.description,
                    "meta": declaration.meta.to_mapping(),
                }
                for declaration in self._declarations.values()
            ]
        ).encode()

        async with ctx.disposables(FileAccess.open(self._path)):
            await File.write(file_contents)
