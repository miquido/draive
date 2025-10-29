from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Literal, final, overload

from haiway import State

from draive.cohere.api import CohereAPI
from draive.cohere.embedding import CohereEmbedding
from draive.embedding import ImageEmbedding, TextEmbedding

__all__ = ("Cohere",)


@final
class Cohere(
    CohereEmbedding,
    CohereAPI,
):
    __slots__ = ("_features",)

    @overload
    def __init__(
        self,
        provider: Literal["bedrock"],
        /,
        *,
        aws_region: str | None = None,
        features: Collection[type[TextEmbedding | ImageEmbedding]] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        provider: Literal["cohere"] = "cohere",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        features: Collection[type[TextEmbedding | ImageEmbedding]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        provider: Literal["cohere", "bedrock"] = "cohere",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        aws_region: str | None = None,
        features: Collection[type[TextEmbedding | ImageEmbedding]] | None = None,
    ) -> None:
        super().__init__(
            provider,
            base_url=base_url,
            api_key=api_key,
            aws_region=aws_region,
        )

        self._features: frozenset[type[State]] = (
            frozenset(features)
            if features is not None
            else frozenset((TextEmbedding, ImageEmbedding))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []

        if TextEmbedding in self._features:
            state.append(self.text_embedding())

        if ImageEmbedding in self._features:
            state.append(self.image_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
