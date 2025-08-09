from collections.abc import Callable, Sequence
from typing import Any, cast, overload

from haiway import AttributePath, State, statemethod

from draive.embedding.types import Embedded, ValueEmbedding
from draive.parameters import DataModel

__all__ = (
    "ImageEmbedding",
    "TextEmbedding",
)


class TextEmbedding(State):
    @overload
    @classmethod
    async def embed(
        cls,
        content: str,
        /,
        **extra: Any,
    ) -> Embedded[str]: ...

    @overload
    async def embed(
        self,
        content: str,
        /,
        **extra: Any,
    ) -> Embedded[str]: ...

    @overload
    @classmethod
    async def embed(
        cls,
        content: Sequence[str],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[str]]: ...

    @overload
    async def embed(
        self,
        content: Sequence[str],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[str]]: ...

    @overload
    @classmethod
    async def embed[Value: DataModel | State](
        cls,
        content: Value,
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str,
        **extra: Any,
    ) -> Embedded[Value]: ...

    @overload
    async def embed[Value: DataModel | State](
        self,
        content: Value,
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str,
        **extra: Any,
    ) -> Embedded[Value]: ...

    @overload
    @classmethod
    async def embed[Value: DataModel | State](
        cls,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @overload
    async def embed[Value: DataModel | State](
        self,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @statemethod
    async def embed[Value: DataModel | State | str](
        self,
        content: Sequence[Value] | Sequence[str] | Value | str,
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]] | Embedded[Value] | Embedded[str]:
        str_selector: Callable[[Value], str] | None
        match attribute:
            case None:
                str_selector = None

            case Callable() as function:  # pyright: ignore[reportUnknownVariableType]
                str_selector = function

            case path:
                assert isinstance(  # nosec: B101
                    path, AttributePath
                ), "Prepare parameter path by using Self._.path.to.property"
                str_selector = cast(AttributePath[Value, str], path).__call__

        match content:
            case [*elements]:
                return await self.embedding(
                    elements,
                    attribute=str_selector,
                    **extra,
                )

            case element:
                return (
                    await self.embedding(
                        [element],
                        attribute=str_selector,
                        **extra,
                    )
                )[0]

    embedding: ValueEmbedding[Any, str]


class ImageEmbedding(State):
    @overload
    @classmethod
    async def embed(
        cls,
        content: bytes,
        /,
        **extra: Any,
    ) -> Embedded[bytes]: ...

    @overload
    async def embed(
        self,
        content: bytes,
        /,
        **extra: Any,
    ) -> Embedded[bytes]: ...

    @overload
    @classmethod
    async def embed(
        cls,
        content: Sequence[bytes],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[bytes]]: ...

    @overload
    async def embed(
        self,
        content: Sequence[bytes],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[bytes]]: ...

    @overload
    @classmethod
    async def embed[Value: DataModel | State](
        cls,
        content: Value,
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes,
        **extra: Any,
    ) -> Embedded[Value]: ...

    @overload
    async def embed[Value: DataModel | State](
        self,
        content: Value,
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes,
        **extra: Any,
    ) -> Embedded[Value]: ...

    @overload
    @classmethod
    async def embed[Value: DataModel | State](
        cls,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @overload
    async def embed[Value: DataModel | State](
        self,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @statemethod
    async def embed[Value: DataModel | State | bytes](
        self,
        content: Sequence[Value] | Sequence[bytes] | Value | bytes,
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[bytes]] | Embedded[Value] | Embedded[bytes]:
        bytes_selector: Callable[[Value], bytes] | None
        match attribute:
            case None:
                bytes_selector = None

            case Callable() as function:  # pyright: ignore[reportUnknownVariableType]
                bytes_selector = function

            case path:
                assert isinstance(  # nosec: B101
                    path, AttributePath
                ), "Prepare parameter path by using Self._.path.to.property"
                bytes_selector = cast(AttributePath[Value, bytes], path).__call__

        match content:
            case [*elements]:
                return await self.embedding(
                    elements,
                    attribute=bytes_selector,
                    **extra,
                )

            case element:
                return (
                    await self.embedding(
                        [element],
                        attribute=bytes_selector,
                        **extra,
                    )
                )[0]

    embedding: ValueEmbedding[Any, bytes]
