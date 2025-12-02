from collections.abc import Callable, Collection, Sequence
from typing import Any, cast, final, overload

from haiway import (
    AttributePath,
    AttributeRequirement,
    State,
    ctx,
    statemethod,
)

from draive.embedding.types import (
    Embedded,
    ValueEmbedding,
    VectorDeleting,
    VectorIndexing,
    VectorSearching,
)
from draive.multimodal import TextContent
from draive.parameters import DataModel
from draive.resources import ResourceContent

__all__ = (
    "ImageEmbedding",
    "TextEmbedding",
    "VectorIndex",
)


class TextEmbedding(State):
    """Text embedding implementation.

    Provides convenience wrappers around a configured ``embedding`` state value so
    callers can embed raw strings or strongly-typed ``DataModel``/``State``
    instances.
    """

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

    @statemethod
    async def embed[Value: DataModel | State | str](
        self,
        content: Value | str,
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str | None = None,
        **extra: Any,
    ) -> Embedded[Value] | Embedded[str]:
        """Embed a single string or model into vector space.

        Parameters
        ----------
        content
            Raw text or a model instance to embed.
        attribute
            Optional attribute selector returning the text to embed when
            ``content`` is a model instance. Accepts an ``AttributePath`` or a
            callable.
        **extra
            Provider-specific keyword arguments forwarded to the underlying
            embedding implementation.

        Returns
        -------
        Embedded[str] or Embedded[Value]
            Embedding result matching the input payload type.
        """
        str_selector: Callable[[Value], str] | None
        if attribute is None:
            str_selector = None

        else:
            assert isinstance(  # nosec: B101
                attribute, AttributePath | Callable
            ), "Prepare parameter path by using Type._.path.to.property"
            str_selector = cast(Callable[[Value], str], attribute)

        return (
            await self.embedding(
                [content],
                attribute=str_selector,
                **extra,
            )
        )[0]

    @overload
    @classmethod
    async def embed_many(
        cls,
        content: Sequence[str],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[str]]: ...

    @overload
    async def embed_many(
        self,
        content: Sequence[str],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[str]]: ...

    @overload
    @classmethod
    async def embed_many[Value: DataModel | State](
        cls,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @overload
    async def embed_many[Value: DataModel | State](
        self,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @statemethod
    async def embed_many[Value: DataModel | State | str](
        self,
        content: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | AttributePath[Value, str] | str | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        """Embed many strings or models in a single batch call.

        Parameters
        ----------
        content
            Iterable of raw texts or model instances to embed.
        attribute
            Optional selector producing text from each model entry.
        **extra
            Provider-specific keyword arguments forwarded to the embedding
            implementation.

        Returns
        -------
        Sequence[Embedded[str]] or Sequence[Embedded[Value]]
            Embedding results in the same order as inputs.
        """
        if not content:
            return ()

        str_selector: Callable[[Value], str] | None
        if attribute is None:
            str_selector = None

        else:
            assert isinstance(  # nosec: B101
                attribute, AttributePath | Callable
            ), "Prepare parameter path by using Type._.path.to.property"
            str_selector = cast(Callable[[Value], str], attribute)

        return await self.embedding(
            content,
            attribute=str_selector,
            **extra,
        )

    embedding: ValueEmbedding[Any, str]


class ImageEmbedding(State):
    """Image embedding implementation

    Provides convenience wrappers around a configured ``embedding`` state value so
    callers can embed raw image data or strongly-typed ``DataModel``/``State``
    instances.
    """

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

    @statemethod
    async def embed[Value: DataModel | State | bytes](
        self,
        content: Sequence[Value] | Sequence[bytes] | Value | bytes,
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes | None = None,
        **extra: Any,
    ) -> Embedded[Value] | Embedded[bytes]:
        """Embed a single image payload or model-bound bytes.

        Parameters
        ----------
        content
            Raw image bytes or a model instance containing them.
        attribute
            Optional selector extracting image bytes from the model.
        **extra
            Provider-specific keyword arguments for the embedding backend.

        Returns
        -------
        Embedded[bytes] or Embedded[Value]
            Embedding result matching the provided payload type.
        """
        bytes_selector: Callable[[Value], bytes] | None
        if attribute is None:
            bytes_selector = None

        else:
            assert isinstance(  # nosec: B101
                attribute, AttributePath | Callable
            ), "Prepare parameter path by using Type._.path.to.property"
            bytes_selector = cast(Callable[[Value], bytes], attribute)

        return (
            await self.embedding(
                [content],
                attribute=bytes_selector,
                **extra,
            )
        )[0]

    @overload
    @classmethod
    async def embed_many(
        cls,
        content: Sequence[bytes],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[bytes]]: ...

    @overload
    async def embed_many(
        self,
        content: Sequence[bytes],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[bytes]]: ...

    @overload
    @classmethod
    async def embed_many[Value: DataModel | State](
        cls,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @overload
    async def embed_many[Value: DataModel | State](
        self,
        content: Sequence[Value],
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes,
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...

    @statemethod
    async def embed_many[Value: DataModel | State | bytes](
        self,
        content: Sequence[Value] | Sequence[bytes],
        /,
        attribute: Callable[[Value], bytes] | AttributePath[Value, bytes] | bytes | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[bytes]]:
        """Embed multiple images in one batch request.

        Parameters
        ----------
        content
            Iterable of raw image bytes or model instances.
        attribute
            Optional selector extracting bytes from each model entry.
        **extra
            Provider-specific keyword arguments forwarded to the backend.

        Returns
        -------
        Sequence[Embedded[bytes]] or Sequence[Embedded[Value]]
            Embedding results aligned with the input ordering.
        """
        if not content:
            return ()

        bytes_selector: Callable[[Value], bytes] | None
        if attribute is None:
            bytes_selector = None

        else:
            assert isinstance(  # nosec: B101
                attribute, AttributePath | Callable
            ), "Prepare parameter path by using Type._.path.to.property"
            bytes_selector = cast(Callable[[Value], bytes], attribute)

        return await self.embedding(
            content,
            attribute=bytes_selector,
            **extra,
        )

    embedding: ValueEmbedding[Any, bytes]


@final
class VectorIndex(State):
    """Vector index/store implementation handling index/search/delete operations."""

    @overload
    @classmethod
    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        cls,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None: ...

    @overload
    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        self,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        self,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None:
        """Index a collection of model instances in the configured vector store.

        Parameters
        ----------
        model
            Data model type being indexed.
        attribute
            Selector describing which value to embed and index for each model.
        values
            Models to index.
        **extra
            Provider-specific keyword arguments forwarded to the indexing backend.
        """
        ctx.record_info(
            event="vector_index.index",
            attributes={
                "model": model.__qualname__,
                "values": len(values),
            },
        )
        return await self.indexing(
            model,
            attribute=attribute,
            values=values,
            **extra,
        )

    @overload
    @classmethod
    async def search[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @overload
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @overload
    @classmethod
    async def search[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @overload
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @statemethod
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str | None = None,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]:
        """Query the vector index for similar items.

        Parameters
        ----------
        model
            Data model type to search within.
        query
            Optional explicit query vector or content; when omitted the provider
            may use stored queries or defaults.
        score_threshold
            Minimum similarity score to include a result. Value from 0.0 to 1.0.
        requirements
            Attribute-level constraints to filter results.
        limit
            Maximum number of results to return.
        **extra
            Provider-specific keyword arguments forwarded to the search backend.

        Returns
        -------
        Sequence[Model]
            Matching model instances ordered by similarity.
        """
        ctx.record_info(
            event="vector_index.search",
            attributes={"model": model.__qualname__},
        )
        results: Sequence[Model] = await self.searching(
            model,
            query=query,
            score_threshold=score_threshold,
            requirements=requirements,
            limit=limit,
            **extra,
        )

        return results

    @overload
    @classmethod
    async def delete[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None: ...

    @overload
    async def delete[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def delete[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None:
        """Remove indexed entries that satisfy given requirements.

        Parameters
        ----------
        model
            Data model type whose entries should be deleted.
        requirements
            Optional attribute constraints selecting entries to drop.
        **extra
            Provider-specific keyword arguments forwarded to the delete backend.
        """
        ctx.record_info(
            event="vector_index.delete",
            attributes={"model": model.__qualname__},
        )
        return await self.deleting(
            model,
            requirements=requirements,
            **extra,
        )

    indexing: VectorIndexing
    searching: VectorSearching
    deleting: VectorDeleting
