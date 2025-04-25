from collections.abc import Iterable, Set
from types import TracebackType
from typing import Any, Literal, final

from google.genai.client import HttpOptionsDict  # pyright: ignore[reportPrivateImportUsage]
from haiway import State

from draive.gemini.api import GeminiAPI
from draive.gemini.embedding import GeminiEmbedding
from draive.gemini.lmm_generation import GeminiLMMGeneration
from draive.gemini.lmm_session import GeminiLMMSession
from draive.gemini.tokenization import GeminiTokenization

__all__ = ("Gemini",)


@final
class Gemini(
    GeminiLMMGeneration,
    GeminiLMMSession,
    GeminiEmbedding,
    GeminiTokenization,
    GeminiAPI,
):
    """
    Access to Gemini services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        api_key: str | None = None,
        http_options: HttpOptionsDict | None = None,
        features: Set[Literal["lmm", "lmm_session", "text_embedding"]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            http_options=http_options,
            **extra,
        )

        self._features: frozenset[Literal["lmm", "lmm_session", "text_embedding"]] = (
            frozenset(features)
            if features is not None
            else frozenset(("lmm", "lmm_session", "text_embedding"))
        )

    async def __aenter__(self) -> Iterable[State]:
        state: list[State] = []
        if "lmm" in self._features:
            state.append(self.lmm())

        if "lmm_session" in self._features:
            state.append(self.lmm_session())

        if "text_embedding" in self._features:
            state.append(self.text_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
