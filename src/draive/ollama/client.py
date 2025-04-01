from collections.abc import Iterable, Set
from types import TracebackType
from typing import Literal, final

from haiway import State

from draive.ollama.api import OllamaAPI
from draive.ollama.embedding import OllamaEmbedding
from draive.ollama.lmm_generation import OllamaLMMGeneration

__all__ = [
    "Ollama",
]


@final
class Ollama(
    OllamaLMMGeneration,
    OllamaEmbedding,
    OllamaAPI,
):
    """
    Access to Ollama services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_disposable_state",)

    def __init__(
        self,
        server_url: str | None = None,
        disposable_state: Set[Literal["lmm", "text_embedding"]] | None = None,
    ) -> None:
        super().__init__(server_url=server_url)

        self._disposable_state: frozenset[Literal["lmm", "text_embedding"]] = (
            frozenset(disposable_state)
            if disposable_state is not None
            else frozenset(("lmm", "text_embedding"))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if "lmm" in self._disposable_state:
            state.append(self.lmm())

        if "text_embedding" in self._disposable_state:
            state.append(self.text_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
