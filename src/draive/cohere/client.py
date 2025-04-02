from collections.abc import Iterable, Set
from types import TracebackType
from typing import Literal, final, overload

from haiway import State

from draive.cohere.api import CohereAPI
from draive.cohere.embedding import CohereEmbedding

__all__ = ("Cohere",)


@final
class Cohere(
    CohereEmbedding,
    CohereAPI,
):
    """
    Access to Cohere services, can be used to prepare various functionalities like embedding.
    """

    __slots__ = ("_disposable_state",)

    @overload
    def __init__(
        self,
        provider: Literal["bedrock"],
        /,
        *,
        timeout: float = 60.0,
        disposable_state: Set[Literal["text_embedding", "image_embedding"]] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        provider: Literal["cohere"] = "cohere",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        disposable_state: Set[Literal["text_embedding", "image_embedding"]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        provider: Literal["cohere", "bedrock"] = "cohere",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        disposable_state: Set[Literal["text_embedding", "image_embedding"]] | None = None,
    ) -> None:
        super().__init__(
            provider,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        self._disposable_state: frozenset[Literal["text_embedding", "image_embedding"]] = (
            frozenset(disposable_state)
            if disposable_state is not None
            else frozenset(("text_embedding",))
        )

    async def __aenter__(self) -> Iterable[State]:
        state: list[State] = []

        if "text_embedding" in self._disposable_state:
            state.append(self.text_embedding())

        if "image_embedding" in self._disposable_state:
            state.append(self.image_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
