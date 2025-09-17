from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Any, final

from haiway import State

from draive.embedding import TextEmbedding
from draive.models import GenerativeModel
from draive.ollama.api import OllamaAPI
from draive.ollama.chat import OllamaChat
from draive.ollama.embedding import OllamaEmbedding

__all__ = ("Ollama",)


@final
class Ollama(
    OllamaChat,
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
        timeout: float | None = None,
        features: Collection[type[GenerativeModel | TextEmbedding]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            server_url=server_url,
            timeout=timeout,
            **extra,
        )

        self._features: frozenset[type[State]] = (
            frozenset(features)
            if features is not None
            else frozenset((GenerativeModel, TextEmbedding))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
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
        await self._deinitialize_client()
