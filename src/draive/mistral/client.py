from collections.abc import Iterable, Set
from types import TracebackType
from typing import Literal, final

from haiway import State

from draive.mistral.api import MistralAPI
from draive.mistral.embedding import MistralEmbedding
from draive.mistral.lmm_generation import MistralLMMGeneration
from draive.mistral.tokenization import MistralTokenization

__all__ = [
    "Mistral",
]


@final
class Mistral(
    MistralLMMGeneration,
    MistralEmbedding,
    MistralTokenization,
    MistralAPI,
):
    """
    Access to Mistral services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_disposable_state",)

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        disposable_state: Set[Literal["lmm", "text_embedding"]] | None = None,
    ) -> None:
        super().__init__(
            server_url=server_url,
            api_key=api_key,
            timeout=timeout,
        )

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
