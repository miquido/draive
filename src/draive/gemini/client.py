from collections.abc import Iterable, Set
from types import TracebackType
from typing import Any, Literal, final

from haiway import State

from draive.gemini.api import GeminiAPI
from draive.gemini.embedding import GeminiEmbedding
from draive.gemini.lmm_generation import GeminiLMMGeneration
from draive.gemini.tokenization import GeminiTokenization

__all__ = [
    "Gemini",
]


@final
class Gemini(
    GeminiLMMGeneration,
    GeminiEmbedding,
    GeminiTokenization,
    GeminiAPI,
):
    """
    Access to Gemini services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_disposable_state",)

    def __init__(
        self,
        api_key: str | None = None,
        disposable_state: Set[Literal["lmm", "text_embedding"]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            **extra,
        )

        self._disposable_state: frozenset[Literal["lmm", "text_embedding"]] = (
            frozenset(disposable_state)
            if disposable_state is not None
            else frozenset(("lmm", "text_embedding"))
        )

    async def __aenter__(self) -> Iterable[State]:
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
        pass
