from collections.abc import Collection, Iterable, Mapping
from types import TracebackType
from typing import Any, final

from haiway import State

from draive.embedding import TextEmbedding
from draive.models import GenerativeModel
from draive.vllm.api import VLLMAPI
from draive.vllm.embedding import VLLMEmbedding
from draive.vllm.messages import VLLMMessages

__all__ = ("VLLM",)


@final
class VLLM(
    VLLMMessages,
    VLLMEmbedding,
    VLLMAPI,
):
    """
    Access to VLLM services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        features: Collection[type[State]] | None = None,
        timeout: float | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            default_headers=default_headers,
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
