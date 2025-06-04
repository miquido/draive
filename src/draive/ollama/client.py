from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Literal, final

from haiway import State

from draive.ollama.api import OllamaAPI
from draive.ollama.embedding import OllamaEmbedding
from draive.ollama.lmm_generation import OllamaLMMGeneration

__all__ = ("Ollama",)


@final
class Ollama(
    OllamaLMMGeneration,
    OllamaEmbedding,
    OllamaAPI,
):
    """
    Access to Ollama services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        server_url: str | None = None,
        features: Collection[Literal["lmm", "text_embedding"]] | None = None,
    ) -> None:
        super().__init__(server_url=server_url)

        self._features: frozenset[Literal["lmm", "text_embedding"]] = (
            frozenset(features) if features is not None else frozenset(("lmm", "text_embedding"))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if "lmm" in self._features:
            state.append(self.lmm())

        if "text_embedding" in self._features:
            state.append(self.text_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
