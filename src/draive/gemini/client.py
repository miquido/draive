from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Any, final

from google.genai.client import HttpOptionsDict  # pyright: ignore[reportPrivateImportUsage]
from haiway import State

from draive.embedding import TextEmbedding
from draive.gemini.api import GeminiAPI
from draive.gemini.embedding import GeminiEmbedding
from draive.gemini.generating import GeminiGenerating
from draive.models import GenerativeModel

__all__ = ("Gemini",)


@final
class Gemini(
    GeminiGenerating,
    GeminiEmbedding,
    GeminiAPI,
):
    """
    Access to Gemini services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        api_key: str | None = None,
        vertexai: bool | None = None,
        http_options: HttpOptionsDict | None = None,
        features: Collection[type[GenerativeModel | TextEmbedding]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            vertexai=vertexai,
            http_options=http_options,
            **extra,
        )

        self._features: frozenset[type[State]] = (
            frozenset(features)
            if features is not None
            else frozenset((GenerativeModel, TextEmbedding))
        )

    async def __aenter__(self) -> Iterable[State]:
        state: list[State] = []
        if GenerativeModel in self._features:
            state.append(self.generative_model())

        if TextEmbedding in self._features:
            state.append(self.text_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
